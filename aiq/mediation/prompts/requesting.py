from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path


async def generate_opening_statement(llm, state) -> str:
    """Generate an opening statement for the requesting party.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary and mediator's opening statement

    Returns:
        The generated opening statement
    """
    # Load the system prompt from file
    system_prompt_file = (
        Path(__file__).parent / "requesting_party_opening_statement.txt"
    )
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create human message with case context and mediator's opening statement
    human_message = HumanMessage(
        content=f"""Please provide an opening statement for this mediation case:

Here is a case summary:
{state.case_summary}

This was the mediator's opening statement:
{state.mediator_opening_statement}

Your opening statement should be professional, constructive, and show openness to resolution while presenting your client's perspective clearly and calmly."""
    )

    # Generate opening statement using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def generate_joint_discussion_response(llm, state) -> str:
    """Generate a response for the requesting party during joint discussion.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary and conversation history

    Returns:
        The generated response
    """
    # Load the system prompt from file
    system_prompt_file = Path(__file__).parent / "requesting_party_joint_discussion.txt"
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create a summary of recent events for context
    recent_messages_summary = "\n".join(
        [
            f"{message.additional_kwargs.get('speaker')}: {message.additional_kwargs.get('summary')}"
            for message in state.messages[-5:]
        ]
    )

    # Create human message with conversation context
    human_message = HumanMessage(
        content=f"""Here is the case summary:
{state.case_summary}

Recent conversation:
{recent_messages_summary}

Please provide your next response as the requesting party. Focus on presenting your client's perspective clearly and engaging constructively with the other party."""
    )

    # Generate response using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def generate_negotiation_requesting_party(llm, state) -> str:
    """Generate a negotiation response for the requesting party.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary and negotiation history

    Returns:
        The generated negotiation response
    """
    # Load the system prompt from file
    system_prompt_file = Path(__file__).parent / "requesting_party_negotiation.txt"
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create a summary of recent negotiation events for context
    recent_messages_summary = "\n".join(
        [
            f"{message.additional_kwargs.get('speaker')}: {message.additional_kwargs.get('summary')}"
            for message in state.messages[-5:]
        ]
    )

    # Create human message with negotiation context
    human_message = HumanMessage(
        content=f"""Here is the case summary:
{state.case_summary}

Recent negotiation:
{recent_messages_summary}

Please provide your next response as the requesting party. Focus on presenting clear proposals and engaging constructively in the negotiation process."""
    )

    # Generate response using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def generate_conclusion_requesting_party(llm, state) -> str:
    """Generate a conclusion response for the requesting party.

    Args:
        llm: The language model to use for generation
        state: The current mediation state containing case summary and conversation history

    Returns:
        The generated conclusion response
    """
    # Load the system prompt from file
    system_prompt_file = Path(__file__).parent / "requesting_party_conclusion.txt"
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_message_content = f.read()
    system_message = SystemMessage(content=system_message_content)

    # Create a summary of recent events for context
    recent_messages_summary = "\n".join(
        [
            f"{message.additional_kwargs.get('speaker')}: {message.additional_kwargs.get('summary')}"
            for message in state.messages[-5:]
        ]
    )

    # Create human message with conversation context
    human_message = HumanMessage(
        content=f"""Here is the case summary:
{state.case_summary}

Recent conversation:
{recent_messages_summary}

Please provide your concluding remarks as the requesting party. Focus on acknowledging any agreements reached, expressing appreciation for the process, and confirming your understanding of next steps."""
    )

    # Generate response using LLM
    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)
