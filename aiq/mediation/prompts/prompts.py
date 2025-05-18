from langchain_core.messages import SystemMessage, HumanMessage

async def generate_summary(llm, content: str, context: str = "") -> str:
    """Generate a concise summary of the given content using the LLM.

    Args:
        llm: The language model to use for summarization
        content: The content to summarize
        context: Optional additional context to consider during summarization

    Returns:
        A concise summary of the content
    """
    system_message = SystemMessage(content="""Your job is to summarize statements made in a mediation competition.
                                   Your summary should be one or two sentences and include a very brief and objective description of what was said.
                                   Be sure to include the speaker's role before your summary.
                                   Do not include any other information, headers, or comments about the content, just include the summary itself.""")

    human_message = HumanMessage(content=f"""Please provide a concise summary of the following statement:

{content}

{context if context else ''}

Provide a brief, objective summary that captures the key points.""")

    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, 'content') else str(response)
