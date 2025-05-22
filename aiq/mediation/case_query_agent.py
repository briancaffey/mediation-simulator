import logging
from pydantic import BaseModel, Field


from aiq.builder.function import Function, FunctionInfo
from aiq.builder.builder import Builder
from aiq.data_models.function import FunctionBaseConfig
from aiq.retriever.milvus.register import MilvusRetrieverConfig
from aiq.cli.register_workflow import register_function
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.agent.react_agent.agent import ReActAgentGraph
from aiq.agent.react_agent.agent import ReActGraphState
from aiq.data_models.component_ref import LLMRef

from langchain_core.messages import SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

logger = logging.getLogger(__name__)


# Input schema for the tool
class CaseQueryInputSchema(BaseModel):
    # case_id: str = Field(description="The ID of the case to query")
    question: str = Field(description="The question to ask about the case")


# Output schema for the tool
class CaseQueryOutputSchema(BaseModel):
    answer: str = Field(description="The answer to the question about the case")


USER_PROMPT = """
Question: {question}
"""


class CaseQueryAgentConfig(FunctionBaseConfig, name="case_query_agent"):
    """ReAct agent that uses the case query tool to answer questions about cases."""

    llm_name: LLMRef = Field(description="The LLM model to use with the ReAct agent")
    verbose: bool = Field(
        default=False, description="Set the verbosity of the agent's logging"
    )
    tool_names: list[str] = Field(
        default=[], description="The names of the tools to use with the ReAct agent"
    )
    description: str = Field(
        default="ReAct agent that answers questions about cases using the case query tool"
    )
    max_iterations: int = Field(
        default=2, description="Maximum number of tool calls before stopping"
    )
    system_prompt: str = Field(
        default="""You are a helpful assistant that either answers questions about a specific case or asks questions about the case.
You have access to a case query tool that can look up information about a case if you need to answer a question about the case.

You will be provided a recent conversation history and you should decide if you need to answer a question about the case or if you need to ask a question about the case.

If you need to answer a question about the case, use the case query tool to get the information you need.

If you need to ask a question about the case, then simple ask the question without using any tools.

Respond to the user as best you can. You may ask the human to use the following tools:

{tools}

You may respond in one of two formats.
Use the following format exactly to ask the human to use a tool:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (if there is no required input, include "Action Input: None")
Observation: wait for the human to respond with the result from the tool, do not assume the response

... (this Thought/Action/Action Input/Observation can repeat N times. If you do not need to use a tool, or after asking the human to use any tools and waiting for the human to respond, you might know the final answer.)
Use the following format once you have the final answer:

Thought: I now know the final answer
Final Answer: the final answer to the original input question""",
        description="System prompt for the ReAct agent",
    )
    retry_parsing_errors: bool = Field(
        default=True, description="Specify retrying when encountering parsing errors"
    )
    max_retries: int = Field(
        default=1,
        description="Set the number of retries before raising a parsing error",
    )


@register_function(config_type=CaseQueryAgentConfig)
async def case_query_agent(config: CaseQueryAgentConfig, builder: Builder):
    """Creates a ReAct agent that uses the case query tool."""
    logger.info("🏗️ Building case query agent")
    logger.info("⚙️ Configuration: %s", config)

    # Get the LLM
    logger.info("🤖 Getting LLM with name: %s", config.llm_name)
    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    logger.info("✅ Successfully initialized LLM")

    # Get the case query tool
    logger.info("🔧 Getting tools with names: %s", config.tool_names)
    tools = builder.get_tools(
        config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    if not tools:
        logger.error("❌ No tools found with names: %s", config.tool_names)
        raise ValueError("Case query tool 'case_document_rag' not found")
    logger.info("✅ Successfully loaded %d tools", len(tools))

    # Create a custom prompt that includes the case_id context
    logger.info("📝 Creating chat prompt template")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(config.system_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )
    logger.info("✅ Prompt template created successfully")

    # Create the ReAct agent
    logger.info(
        "🤖 Initializing ReAct agent with max_iterations=%d, verbose=%s",
        config.max_iterations,
        config.verbose,
    )

    # This is the prompt - (ReAct Agent prompt)
    react_agent_prompt = ChatPromptTemplate(
        [
            ("system", config.system_prompt),
            ("user", USER_PROMPT),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )

    agent = ReActAgentGraph(
        llm=llm,
        prompt=react_agent_prompt,
        tools=tools,
        detailed_logs=config.verbose,
        retry_parsing_errors=config.retry_parsing_errors,
        max_retries=config.max_retries,
    )

    # Build the agent graph
    logger.info("🔄 Building agent graph")
    graph = await agent.build_graph()
    logger.info("✅ Agent graph built successfully")

    async def _inner(query: str) -> str:
        try:
            # Initialize the state with the case_id context
            state = ReActGraphState(
                messages=[
                    SystemMessage(content=config.system_prompt),
                    HumanMessage(content=query),
                ],
            )
            logger.info(
                "📨 Initial state created with %d messages", len(state.messages)
            )

            # Run the agent
            logger.info(
                "⚡ Invoking agent graph with recursion limit: %d",
                config.max_iterations + 1,
            )
            result = await graph.ainvoke(state, config={"recursion_limit": 4})

            # Extract the final answer from the result
            if isinstance(result, dict) and "messages" in result:
                final_answer = result["messages"][-1].content
            else:
                # If the result is not in the expected format, try to get the last message
                final_answer = str(result)

            logger.info(
                "📤 Returning final answer (length: %d characters)", len(final_answer)
            )
            return final_answer
        except Exception as ex:
            logger.exception(
                "💥 Case query agent failed with exception: %s", ex, exc_info=True
            )
            if config.verbose:
                logger.info("ℹ️ Returning detailed error due to verbose mode")
                return str(ex)
            logger.info("⚠️ Returning generic error message")
            return (
                "I seem to be having a problem answering your question about the case."
            )

    logger.info("🚀 Yielding function info")
    yield FunctionInfo.from_fn(fn=_inner, description=config.description)
