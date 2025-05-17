"""
This file defines the case_generation workflow.

It creates a description of a case for use in a mediation simulation

"""


import logging
import os
import random
import string
import json
from pathlib import Path
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MediationWorkflowConfig(FunctionBaseConfig, name="mediation"):
    # Add your custom configuration parameters here
    llm: LLMRef = "nim_llm"
    data_dir: str = "./data"


@register_function(config_type=MediationWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def case_generation_workflow(config: MediationWorkflowConfig, builder: Builder):
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph
    from langgraph.graph import END

    import uuid
    import datetime
    from typing import List, Optional, Dict, Literal, Annotated
    from pydantic import BaseModel, Field

    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.prompts import MessagesPlaceholder
    from langgraph.graph import StateGraph, END
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.graphviz import save_workflow_visualization
    from utils.serialize import serialize_pydantic
    from utils.yaml import save_state_to_yaml

    logger.info("Getting LLM with name: %s", config.llm)
    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # use separate LLMs to simulate LLMs representing each party
    # TODO: move these to tool configuration
    # requesting_party_llm = await builder.get_llm(llm_name=config.requesting_party_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # responding_party_llm = await builder.get_llm(llm_name=config.responding_party_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("LLM initialized: %s", llm)

    class Party(BaseModel): # Using BaseModel to allow future extension if needed
        name: Literal["MEDIATOR", "REQUESTING_PARTY", "RESPONDING_PARTY", "CLERK_SYSTEM"]

        def __str__(self):
            return self.name

        def __hash__(self): # Make it hashable for dict keys if needed
            return hash(self.name)

        def __eq__(self, other):
            if isinstance(other, Party):
                return self.name == other.name
            return False


    class MediationPhase(BaseModel):
        name: Literal[
            "OPENING_STATEMENTS",
            "JOINT_DISCUSSION_INFO_GATHERING",
            "CAUCUSES",
            "NEGOTIATION_BARGAINING",
            "CONCLUSION_CLOSING_STATEMENTS",
            "ENDED" # Terminal phase
        ]
        def __str__(self):
            return self.name

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            if isinstance(other, MediationPhase):
                return self.name == other.name
            return False

    # Define phase instances
    PHASE_OPENING = MediationPhase(name="OPENING_STATEMENTS")
    PHASE_JOINT_DISCUSSION = MediationPhase(name="JOINT_DISCUSSION_INFO_GATHERING")
    PHASE_CAUCUS = MediationPhase(name="CAUCUSES")
    PHASE_NEGOTIATION = MediationPhase(name="NEGOTIATION_BARGAINING")
    PHASE_CONCLUSION = MediationPhase(name="CONCLUSION_CLOSING_STATEMENTS")
    PHASE_ENDED = MediationPhase(name="ENDED")

    # --- Pydantic Models for State ---

    class MediationEvent(BaseModel):
        event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
        mediation_phase: MediationPhase
        speaker: Party
        content: str
        summary: str
        token_count: int = 0 # Placeholder for now

    class MediationState(BaseModel):
        """State for the mediation workflow"""

        # case state
        case_id: str = Field(default_factory=lambda: "case-" + str(uuid.uuid4())[:8])
        case_summary: str = ""
        requesting_party_confidential_info: str = ""
        responding_party_confidential_info: str = ""

        # New fields for opening statements
        mediator_opening_statement: str = ""
        requesting_party_opening_statement: str = ""
        responding_party_opening_statement: str = ""

        current_phase: MediationPhase = Field(default=PHASE_OPENING)
        events: List[MediationEvent] = Field(default_factory=list)

        turn_number: int = 0
        total_tokens_spoken: int = 0

        # For clerk processing
        last_utterance_content: Optional[str] = None
        last_utterance_speaker: Optional[Party] = None

        # For clerk decision-making
        next_speaker_candidate: Optional[Party] = None

        is_in_caucus: bool = False
        caucus_party: Optional[Party] = None

        # Configuration for phase transitions
        max_turns_per_phase: Dict[MediationPhase, int] = Field(default_factory=lambda: {
            PHASE_OPENING: 3,
            PHASE_JOINT_DISCUSSION: 10,
            PHASE_CAUCUS: 6,
            PHASE_NEGOTIATION: 15,
            PHASE_CONCLUSION: 3,
        })
        turns_in_current_phase: int = 0

        agreement_reached: Optional[bool] = None
        agreement_terms: Optional[str] = None

        # LangGraph messages
        messages: List[BaseMessage] = Field(default_factory=list)

        # Helper to add a new event based on last utterance
        def finalize_and_add_event(self, summary: str, token_count: int):
            if self.last_utterance_content and self.last_utterance_speaker:
                event = MediationEvent(
                    mediation_phase=self.current_phase,
                    speaker=self.last_utterance_speaker,
                    content=self.last_utterance_content,
                    summary=summary,
                    token_count=token_count,
                )
                self.events.append(event)
                self.total_tokens_spoken += token_count
                self.last_utterance_content = None
                return event
            return None

    def initial(state: MediationState):
        """Load the parts of the case data from data_dir based on the case_id that is passed in"""
        case_dir = Path(config.data_dir) / state.case_id
        if not case_dir.exists():
            raise ValueError(f"Case directory {case_dir} does not exist")

        # Load the case data from the case_dir
        case_file = case_dir / "case_generation_state.yaml"
        if not case_file.exists():
            raise ValueError(f"Case file {case_file} does not exist")

        import yaml
        with open(case_file, 'r', encoding='utf-8') as f:
            case_data = yaml.safe_load(f)

        # Set the case summary from the basic case information
        state.case_summary = case_data.get('basic_case_information', '')

        return state

    async def generate_clerk_decision(llm, state: MediationState) -> str:
        """Generate a decision about who should speak next based on the conversation history.

        Args:
            llm: The language model to use for decision making
            state: The current mediation state

        Returns:
            A string indicating who should speak next: "MEDIATOR", "REQUESTING_PARTY", or "RESPONDING_PARTY"
        """
        # Create a summary of the conversation history
        conversation_history = "\n".join([
            f"{event.speaker.name}: {event.summary}"
            for event in state.events[-5:]  # Look at last 5 events for context
        ])

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
- Ensure each party has a chance to make closing remarks"""
        }

        system_message = SystemMessage(content=f"""You are a mediation clerk responsible for managing the flow of conversation in a legal mediation session.
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

Do not include any other text, explanations, or formatting in your response.""")

        human_message = HumanMessage(content=f"""Current mediation phase: {state.current_phase.name}
Current turn number: {state.turn_number}
Turns in current phase: {state.turns_in_current_phase}

Recent conversation history:
{conversation_history}

Based on this context, who should speak next? Respond with exactly one of: "MEDIATOR", "REQUESTING_PARTY", or "RESPONDING_PARTY".""")

        messages = [system_message, human_message]
        response = await llm.ainvoke(messages)
        decision = response.content.strip().upper()

        # Validate the response
        valid_decisions = {"MEDIATOR", "REQUESTING_PARTY", "RESPONDING_PARTY"}
        if decision not in valid_decisions:
            logger.warning(f"Invalid clerk decision: {decision}. Defaulting to MEDIATOR.")
            decision = "MEDIATOR"

        return decision

    async def clerk_node(state: MediationState):
        """The clerk node decides who speaks next and whether to end the mediation"""
        logger.info(f"[CLERK]: Current phase: {state.current_phase}, Turn: {state.turn_number}")

        # Check if we should end the mediation
        if state.current_phase == PHASE_ENDED:
            return state

        # Handle opening phase
        if state.current_phase == PHASE_OPENING:
            # Check opening statements in order: mediator, requesting party, responding party
            if not state.mediator_opening_statement:
                state.next_speaker_candidate = Party(name="MEDIATOR")
            elif not state.requesting_party_opening_statement:
                state.next_speaker_candidate = Party(name="REQUESTING_PARTY")
            elif not state.responding_party_opening_statement:
                state.next_speaker_candidate = Party(name="RESPONDING_PARTY")
            else:
                # All opening statements are complete, transition to joint discussion
                state.current_phase = PHASE_JOINT_DISCUSSION
                state.turns_in_current_phase = 0
                state.next_speaker_candidate = Party(name="MEDIATOR")  # Mediator typically starts joint discussion
            return state

        # For all other phases, use the LLM to decide the next speaker
        # Check if we've reached max turns for the current phase
        if state.turns_in_current_phase >= state.max_turns_per_phase[state.current_phase]:
            # Transition to next phase
            if state.current_phase == PHASE_JOINT_DISCUSSION:
                # state.current_phase = PHASE_CAUCUS
                # state.is_in_caucus = True
                # state.caucus_party = Party(name="REQUESTING_PARTY")
                state.current_phase = PHASE_CAUCUS
            elif state.current_phase == PHASE_CAUCUS:
                state.current_phase = PHASE_NEGOTIATION
                state.is_in_caucus = False
                state.caucus_party = None
            elif state.current_phase == PHASE_NEGOTIATION:
                state.current_phase = PHASE_CONCLUSION
            elif state.current_phase == PHASE_CONCLUSION:
                state.current_phase = PHASE_ENDED
                return state

            state.turns_in_current_phase = 0
            state.next_speaker_candidate = Party(name="MEDIATOR")  # Mediator typically starts new phases
            return state

        # Get the next speaker from the LLM
        next_speaker = await generate_clerk_decision(llm, state)

        # Additional validation for caucus phase
        if state.current_phase == PHASE_CAUCUS:
            if next_speaker not in ["MEDIATOR", state.caucus_party.name]:
                next_speaker = "MEDIATOR"  # Default to mediator if invalid choice

        state.next_speaker_candidate = Party(name=next_speaker)

        # Increment counters
        state.turns_in_current_phase += 1
        state.turn_number += 1

        return state

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

    async def mediator_node(state: MediationState):
        """The mediator node generates the mediator's response"""
        logger.info("Mediator node called")
        state.last_utterance_speaker = Party(name="MEDIATOR")

        if state.current_phase == PHASE_OPENING and not state.mediator_opening_statement:
            # Create system message for mediator role
            system_prompt_file = Path(__file__).parent / "prompts" / "mediator_opening_statement.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_message_content = f.read()
            system_message = SystemMessage(content=system_message_content)

            # Create human message with case context
            human_message = HumanMessage(content=f"""Please provide an opening statement for this mediation case:

{state.case_summary}

Your opening statement should welcome the parties, establish your role as a neutral facilitator, explain the mediation process, and set expectations for a productive discussion. Reference specific elements from the case to show you understand the context.""")

            # Generate opening statement using LLM
            messages = [system_message, human_message]
            response = await llm.ainvoke(messages)
            opening_statement = response.content if hasattr(response, 'content') else str(response)

            state.mediator_opening_statement = opening_statement
            state.last_utterance_content = opening_statement

            # Generate summary of the opening statement
            summary = await generate_summary(llm, opening_statement, "This is a mediator's opening statement.")

            # Create MediationEvent for the opening statement
            event = MediationEvent(
                mediation_phase=state.current_phase,
                speaker=state.last_utterance_speaker,
                content=opening_statement,
                summary=summary,
                token_count=len(opening_statement.split())  # Rough estimate of tokens
            )
            state.events.append(event)
        elif state.current_phase == PHASE_JOINT_DISCUSSION:
            # Create system message for mediator role in joint discussion
            system_prompt_file = Path(__file__).parent / "prompts" / "mediator_joint_discussion.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_message_content = f.read()
            system_message = SystemMessage(content=system_message_content)

            # Create a summary of recent events for context
            recent_events = "\n".join([
                f"{event.speaker.name}: {event.summary}"
                for event in state.events[-5:]  # Look at last 5 events for context
            ])

            # Create human message with conversation context
            human_message = HumanMessage(content=f"""Here is the case summary:
{state.case_summary}

Summary of recent conversation:
{recent_events}

Please provide your next response as the mediator. Focus on guiding the discussion, helping parties understand each other, and moving toward resolution.""")

            # Generate response using LLM
            messages = [system_message, human_message]
            response = await llm.ainvoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)

            state.last_utterance_content = response_content

            # Generate summary of the response
            summary = await generate_summary(llm, response_content, "This is a mediator's response during joint discussion.")

            # Create MediationEvent for the response
            event = MediationEvent(
                mediation_phase=state.current_phase,
                speaker=state.last_utterance_speaker,
                content=response_content,
                summary=summary,
                token_count=len(response_content.split())  # Rough estimate of tokens
            )
            state.events.append(event)
        else:
            state.last_utterance_content = f"Mediator speaking on turn {state.turn_number}."

        return state

    async def requesting_party_node(state: MediationState):
        """The requesting party node generates responses"""
        logger.info("Requesting party node called")
        state.last_utterance_speaker = Party(name="REQUESTING_PARTY")

        if state.current_phase == PHASE_OPENING and not state.requesting_party_opening_statement:
            # Create system message for requesting party role
            system_prompt_file = Path(__file__).parent / "prompts" / "requesting_party_opening_statement.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_message_content = f.read()
            system_message = SystemMessage(content=system_message_content)

            # Create human message with case context and mediator's opening statement
            human_message = HumanMessage(content=f"""Please provide an opening statement for this mediation case:

Here is a case summary:
{state.case_summary}

This was the mediator's opening statement:
{state.mediator_opening_statement}

Your opening statement should be professional, constructive, and show openness to resolution while presenting your client's perspective clearly and calmly.""")

            # Generate opening statement using LLM
            messages = [system_message, human_message]
            response = await llm.ainvoke(messages)
            opening_statement = response.content if hasattr(response, 'content') else str(response)

            state.requesting_party_opening_statement = opening_statement
            state.last_utterance_content = opening_statement

            # Generate summary of the opening statement
            summary = await generate_summary(llm, opening_statement, "This is a requesting party's opening statement.")

            # Create MediationEvent for the opening statement
            event = MediationEvent(
                mediation_phase=state.current_phase,
                speaker=state.last_utterance_speaker,
                content=opening_statement,
                summary=summary,
                token_count=len(opening_statement.split())  # Rough estimate of tokens
            )
            state.events.append(event)
        elif state.current_phase == PHASE_JOINT_DISCUSSION:
            # Create system message for requesting party role in joint discussion
            system_prompt_file = Path(__file__).parent / "prompts" / "requesting_party_joint_discussion.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_message_content = f.read()
            system_message = SystemMessage(content=system_message_content)

            # Create a summary of recent events for context
            recent_events = "\n".join([
                f"{event.speaker.name}: {event.summary}"
                for event in state.events[-5:]  # Look at last 5 events for context
            ])

            # Create human message with conversation context
            human_message = HumanMessage(content=f"""Here is the case summary:
{state.case_summary}

Recent conversation:
{recent_events}

Please provide your next response as the requesting party. Focus on presenting your client's perspective clearly and engaging constructively with the other party.""")

            # Generate response using LLM
            messages = [system_message, human_message]
            response = await llm.ainvoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)

            state.last_utterance_content = response_content

            # Generate summary of the response
            summary = await generate_summary(llm, response_content, "This is a requesting party's response during joint discussion.")

            # Create MediationEvent for the response
            event = MediationEvent(
                mediation_phase=state.current_phase,
                speaker=state.last_utterance_speaker,
                content=response_content,
                summary=summary,
                token_count=len(response_content.split())  # Rough estimate of tokens
            )
            state.events.append(event)
        else:
            state.last_utterance_content = f"Requesting party speaking on turn {state.turn_number}."

        return state

    async def responding_party_node(state: MediationState):
        """The responding party node generates responses"""
        logger.info("Responding party node called")
        state.last_utterance_speaker = Party(name="RESPONDING_PARTY")

        if state.current_phase == PHASE_OPENING and not state.responding_party_opening_statement:
            # Create system message for responding party role
            system_prompt_file = Path(__file__).parent / "prompts" / "responding_party_opening_statement.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_message_content = f.read()
            system_message = SystemMessage(content=system_message_content)

            # Create human message with case context, mediator's opening statement, and requesting party's opening statement
            human_message = HumanMessage(content=f"""Please provide an opening statement for this mediation case:

Here is the case summary:
{state.case_summary}

This was the mediator's opening statement:
{state.mediator_opening_statement}

This was the requesting party's opening statement:
{state.requesting_party_opening_statement}

Your opening statement should be professional, constructive, and show openness to resolution while presenting your client's perspective clearly and calmly.
You should acknowledge the requesting party's statement while maintaining your own position.""")

            # Generate opening statement using LLM
            messages = [system_message, human_message]
            response = await llm.ainvoke(messages)
            opening_statement = response.content if hasattr(response, 'content') else str(response)

            state.responding_party_opening_statement = opening_statement
            state.last_utterance_content = opening_statement

            # Generate summary of the opening statement
            summary = await generate_summary(llm, opening_statement, "This is a responding party's opening statement.")

            # Create MediationEvent for the opening statement
            event = MediationEvent(
                mediation_phase=state.current_phase,
                speaker=state.last_utterance_speaker,
                content=opening_statement,
                summary=summary,
                token_count=len(opening_statement.split())  # Rough estimate of tokens
            )
            state.events.append(event)
        elif state.current_phase == PHASE_JOINT_DISCUSSION:
            # Create system message for responding party role in joint discussion
            system_prompt_file = Path(__file__).parent / "prompts" / "responding_party_joint_discussion.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                system_message_content = f.read()
            system_message = SystemMessage(content=system_message_content)

            # Create a summary of recent events for context
            recent_events = "\n".join([
                f"{event.speaker.name}: {event.summary}"
                for event in state.events[-5:]  # Look at last 5 events for context
            ])

            # Create human message with conversation context
            human_message = HumanMessage(content=f"""Here is the case summary:
{state.case_summary}

Recent conversation:
{recent_events}

Please provide your next response as the responding party. Focus on presenting your client's perspective clearly and engaging constructively with the requesting party.""")

            # Generate response using LLM
            messages = [system_message, human_message]
            response = await llm.ainvoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)

            state.last_utterance_content = response_content

            # Generate summary of the response
            summary = await generate_summary(llm, response_content, "This is a responding party's response during joint discussion.")

            # Create MediationEvent for the response
            event = MediationEvent(
                mediation_phase=state.current_phase,
                speaker=state.last_utterance_speaker,
                content=response_content,
                summary=summary,
                token_count=len(response_content.split())  # Rough estimate of tokens
            )
            state.events.append(event)
        else:
            state.last_utterance_content = f"Responding party speaking on turn {state.turn_number}."

        return state

    workflow = StateGraph(MediationState)

    # Add nodes
    workflow.add_node("initial", RunnableLambda(initial))
    workflow.add_node("clerk", RunnableLambda(clerk_node))
    workflow.add_node("mediator", RunnableLambda(mediator_node))
    workflow.add_node("requesting_party", RunnableLambda(requesting_party_node))
    workflow.add_node("responding_party", RunnableLambda(responding_party_node))

    # Set entry point
    workflow.set_entry_point("initial")

    # Add edges
    workflow.add_edge("initial", "clerk")
    workflow.add_edge("mediator", "clerk")
    workflow.add_edge("requesting_party", "clerk")
    workflow.add_edge("responding_party", "clerk")

    # Add conditional edge from clerk
    workflow.add_conditional_edges(
        "clerk",
        lambda x: "end" if x.current_phase == PHASE_ENDED else {
            "MEDIATOR": "mediator",
            "REQUESTING_PARTY": "requesting_party",
            "RESPONDING_PARTY": "responding_party"
        }[x.next_speaker_candidate.name]
    )

    # Compile the workflow with a higher recursion limit
    app = workflow.compile(config={"recursion_limit": 1000})  # Set a high recursion limit

    # Save workflow visualization
    save_workflow_visualization(app)

    async def _response_fn(input_message: str = None) -> str:
        logger.debug("Starting mediation workflow execution")

        case_id = input_message
        if not case_id:
            raise ValueError("Case ID is required, please provide case ID via --input")

        # Initialize the state with the required fields
        initial_state = MediationState(
            case_id=case_id,
            events=[],  # Explicitly initialize empty events list
            case_summary="",  # Initialize empty case summary
            current_phase=PHASE_OPENING,  # Set initial phase
            turn_number=0,  # Initialize turn counter
            total_tokens_spoken=0,  # Initialize token counter
            turns_in_current_phase=0  # Initialize phase turn counter
        )

        # Run the workflow
        output = (await app.ainvoke(initial_state))

        # Create the directory structure
        case_dir = Path(config.data_dir) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Convert the LangGraph output back to our Pydantic model
        state_dict = serialize_pydantic(output)

        # Save the state to YAML
        save_state_to_yaml(state_dict, str(case_dir), "mediation_state")

        return output.get("case_id")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up case_generation workflow.")
