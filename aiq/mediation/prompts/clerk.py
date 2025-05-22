from langchain_core.messages import SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

# Define phase instances
PHASE_OPENING = "OPENING_STATEMENTS"
PHASE_JOINT_DISCUSSION = "JOINT_DISCUSSION_INFO_GATHERING"
PHASE_CAUCUS = "CAUCUSES"
PHASE_NEGOTIATION = "NEGOTIATION_BARGAINING"
PHASE_CONCLUSION = "CONCLUSION_CLOSING_STATEMENTS"
PHASE_ENDED = "ENDED"


async def generate_clerk_decision(llm, state) -> str:
    """Generate a decision about who should speak next based on the conversation history.

    Args:
        llm: The language model to use for decision making
        state: The current mediation state

    Returns:
        A string indicating who should speak next: "MEDIATOR", "REQUESTING_PARTY", or "RESPONDING_PARTY"
    """
    # Create a summary of the conversation history
    conversation_history = "\n".join(
        [
            f"{event.speaker.name}: {event.summary}"
            for event in state.events[-5:]  # Look at last 5 events for context
        ]
    )

    # Add constraints based on the current phase
    phase_constraints = {
        PHASE_JOINT_DISCUSSION: """In joint discussion:
- The mediator should not speak more than twice in a row
- After the mediator speaks, a party should speak next
- Parties should have roughly equal speaking opportunities""",
        PHASE_CAUCUS: """In caucus:
- Only the mediator and the party in caucus should speak
- Alternate between mediator and the party""",
        PHASE_NEGOTIATION: """In negotiation:
- The mediator should speak less frequently
- Parties should negotiate directly with each other
- The mediator should only speak to guide or clarify""",
        PHASE_CONCLUSION: """In conclusion:
- Follow a structured order: mediator → requesting party → responding party → mediator
- Ensure each party has a chance to make closing remarks""",
    }

    system_message = SystemMessage(
        content=f"""You are a mediation clerk responsible for managing the flow of conversation in a legal mediation session.
Your role is to decide who should speak next based on the conversation history and the current phase of mediation.

Consider the following factors:
1. The current phase of mediation (opening statements, joint discussion, caucus, negotiation, or conclusion)
2. Who has spoken recently and what they said
3. Whether a party needs to respond to a specific point
4. Whether the mediator needs to guide the conversation
5. The natural flow of discussion and turn-taking

{phase_constraints.get(state.current_phase, '')}

You must respond with EXACTLY one of these three options:
- "MEDIATOR"
- "REQUESTING_PARTY"
- "RESPONDING_PARTY"

Do not include any other text, explanations, or formatting in your response."""
    )

    human_message = HumanMessage(
        content=f"""Current mediation phase: {state.current_phase.name}
Current turn number: {state.turn_number}
Turns in current phase: {state.turns_in_current_phase}

Recent conversation history:
{conversation_history}

Based on this context, who should speak next? Respond with exactly one of: "MEDIATOR", "REQUESTING_PARTY", or "RESPONDING_PARTY"."""
    )

    messages = [system_message, human_message]
    response = await llm.ainvoke(messages)
    decision = response.content.strip().upper()

    # Validate the response
    valid_decisions = {"MEDIATOR", "REQUESTING_PARTY", "RESPONDING_PARTY"}
    if decision not in valid_decisions:
        logger.warning(f"Invalid clerk decision: {decision}. Defaulting to MEDIATOR.")
        decision = "MEDIATOR"

    return decision
