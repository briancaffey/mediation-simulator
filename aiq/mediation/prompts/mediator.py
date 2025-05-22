from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path


async def generate_opening_statement(llm, state) -> str:
    """Generate an opening statement for the mediator.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary

    Returns:
        The generated opening statement
    """
    # Load the system prompt from file
    system_prompt_file = Path(__file__).parent / "mediator_opening_statement.txt"
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create human message with case context
    human_message = HumanMessage(
        content=f"""Please provide an opening statement for this mediation case:

{state.case_summary}

Your opening statement should welcome the parties, establish your role as a neutral facilitator, explain the mediation process, and set expectations for a productive discussion. Reference specific elements from the case to show you understand the context."""
    )

    # Generate opening statement using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def generate_joint_discussion_response(llm, state) -> str:
    """Generate a response for the mediator during joint discussion.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary and conversation history

    Returns:
        The generated response
    """
    # Load the system prompt from file
    system_prompt_file = Path(__file__).parent / "mediator_joint_discussion.txt"
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create a summary of recent events for context
    recent_events = "\n".join(
        [
            f"{event.speaker.name}: {event.summary}"
            for event in state.events[-5:]  # Look at last 5 events for context
        ]
    )

    # Create human message with conversation context
    human_message = HumanMessage(
        content=f"""Here is the case summary:
{state.case_summary}

Summary of recent conversation:
{recent_events}

Please provide your next response as the mediator. Focus on guiding the discussion, helping parties understand each other, and moving toward resolution."""
    )

    # Generate response using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def generate_negotiation_mediator(llm, state) -> str:
    """Generate a response for the mediator during the negotiation phase.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary and conversation history

    Returns:
        The generated response
    """
    # Load the system prompt from file
    system_prompt_file = Path(__file__).parent / "mediator_negotiation.txt"
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create a summary of recent events for context
    recent_events = "\n".join(
        [
            f"{event.speaker.name}: {event.summary}"
            for event in state.events[-5:]  # Look at last 5 events for context
        ]
    )

    # Create human message with conversation context
    human_message = HumanMessage(
        content=f"""Here is the case summary:
{state.case_summary}

Summary of recent conversation:
{recent_events}

Please provide your next response as the mediator. Focus on guiding the negotiation process, helping parties develop and evaluate concrete proposals, and moving toward a mutually acceptable agreement."""
    )

    # Generate response using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)
