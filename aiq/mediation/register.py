"""
This file defines the mediation tools and workflows.

It creates a description of a case for use in a mediation simulation

"""

import logging
from pathlib import Path

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

# Configure logging to suppress warnings from async_otel_listener
logging.getLogger("aiq.observability.async_otel_listener").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class MediationWorkflowConfig(FunctionBaseConfig, name="mediation"):
    # Add your custom configuration parameters here
    llm: LLMRef = "nim_llm"
    data_dir: str = "./data"


@register_function(
    config_type=MediationWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def case_generation_workflow(config: MediationWorkflowConfig, builder: Builder):
    from langchain_core.messages import BaseMessage, SystemMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import END, StateGraph

    import uuid
    import datetime
    from typing import List, Optional, Dict, Literal
    from pydantic import BaseModel, Field

    from langchain_core.messages import BaseMessage, HumanMessage
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from utils.graphviz import save_workflow_visualization
    from utils.serialize import serialize_pydantic
    from utils.yaml import save_state_to_yaml
    from .prompts.prompts import generate_summary
    from .prompts.clerk import generate_clerk_decision
    from .prompts.responding import (
        generate_opening_statement as generate_responding_opening_statement,
        generate_joint_discussion_response as generate_responding_joint_discussion,
        generate_negotiation_responding_party as generate_negotiation_responding_party,
        generate_conclusion_responding_party as generate_conclusion_responding_party,
    )
    from .prompts.mediator import (
        generate_opening_statement as generate_mediator_opening_statement,
        generate_joint_discussion_response as generate_mediator_joint_discussion,
        generate_negotiation_mediator as generate_negotiation_mediator,
        generate_mediator_conclusion as generate_mediator_conclusion,
    )
    from .prompts.requesting import (
        generate_opening_statement as generate_requesting_opening_statement,
        generate_joint_discussion_response as generate_requesting_joint_discussion,
        generate_negotiation_requesting_party as generate_negotiation_requesting_party,
        generate_conclusion_requesting_party as generate_conclusion_requesting_party,
    )

    logger.info("🤖 Getting LLM with name: %s", config.llm)
    llm = await builder.get_llm(
        llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    # use separate LLMs to simulate LLMs representing each party
    # TODO: move these to tool configuration
    # requesting_party_llm = await builder.get_llm(llm_name=config.requesting_party_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # responding_party_llm = await builder.get_llm(llm_name=config.responding_party_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    logger.info("✅ LLM initialized: %s", llm)

    # Get the case query agent
    case_query_agent = builder.get_function("case_query_agent")

    class Party(BaseModel):  # Using BaseModel to allow future extension if needed
        name: Literal[
            "MEDIATOR", "REQUESTING_PARTY", "RESPONDING_PARTY", "CLERK_SYSTEM"
        ]

        def __str__(self):
            return self.name

        def __hash__(self):  # Make it hashable for dict keys if needed
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
            "ENDED",  # Terminal phase
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
        token_count: int = 0  # Placeholder for now

    class MediationState(BaseModel):
        """State for the mediation workflow"""

        # case state
        case_id: str = Field(default_factory=lambda: "case-" + str(uuid.uuid4())[:8])
        case_title: str = ""
        case_summary: str = ""

        # company names
        requesting_party_company: str = ""
        requesting_party_representative: str = ""
        responding_party_company: str = ""
        responding_party_representative: str = ""

        # confidential info
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
        # TODO: increase phase length. Temporarily using shortened phases for debugging
        max_turns_per_phase: Dict[MediationPhase, int] = Field(
            default_factory=lambda: {
                PHASE_OPENING: 3,
                PHASE_JOINT_DISCUSSION: 5,
                PHASE_CAUCUS: 2,
                PHASE_NEGOTIATION: 5,
                PHASE_CONCLUSION: 3,
            }
        )
        turns_in_current_phase: int = 0

        # conclusion state
        requesting_party_conclusion: str = ""
        responding_party_conclusion: str = ""
        mediator_conclusion_settlement: str = ""

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

    async def initial(state: MediationState):
        """Load the parts of the case data from data_dir based on the case_id that is passed in"""
        case_dir = Path(config.data_dir) / state.case_id
        if not case_dir.exists():
            raise ValueError(f"Case directory {case_dir} does not exist")

        # Load the case data from the case_dir
        case_file = case_dir / "case_generation_state.yaml"
        if not case_file.exists():
            raise ValueError(f"Case file {case_file} does not exist")

        import yaml

        with open(case_file, "r", encoding="utf-8") as f:
            case_data = yaml.safe_load(f)

        # Set the case summary from the basic case information
        state.case_summary = case_data.get("basic_case_information", "")
        state.case_title = case_data.get("case_title", "")

        # Set the company names
        state.requesting_party_company = case_data.get("requesting_party_company", "")
        state.requesting_party_representative = case_data.get("requesting_party_representative", "")
        state.responding_party_company = case_data.get("responding_party_company", "")
        state.responding_party_representative = case_data.get("responding_party_representative", "")

        return state

    async def clerk_node(state: MediationState):
        """The clerk node decides who speaks next and whether to end the mediation"""
        logger.info(
            f"👤 [CLERK]: Current phase: {state.current_phase}, Turn: {state.turn_number}"
        )

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
                state.next_speaker_candidate = Party(
                    name="MEDIATOR"
                )  # Mediator typically starts joint discussion
            return state

        # # Additional validation for caucus phase
        # if state.current_phase == PHASE_CAUCUS:
        #     if not state.is_in_caucus or not state.caucus_party:
        #         # Initialize caucus if not already done
        #         state.is_in_caucus = True
        #         state.caucus_party = Party(name="REQUESTING_PARTY")
        #         next_speaker = "MEDIATOR"  # Start with mediator
        #     elif next_speaker not in ["MEDIATOR", state.caucus_party.name]:
        #         next_speaker = "MEDIATOR"  # Default to mediator if invalid choice

        # Handle conclusion phase
        if state.current_phase == PHASE_CONCLUSION:
            if not state.requesting_party_conclusion:
                state.next_speaker_candidate = Party(name="REQUESTING_PARTY")
            elif not state.responding_party_conclusion:
                state.next_speaker_candidate = Party(name="RESPONDING_PARTY")
            elif not state.mediator_conclusion_settlement:
                state.next_speaker_candidate = Party(name="MEDIATOR")
            else:
                state.current_phase = PHASE_ENDED
            return state

        # For all other phases, use the LLM to decide the next speaker
        # Check if we've reached max turns for the current phase
        if (
            state.turns_in_current_phase
            >= state.max_turns_per_phase[state.current_phase]
        ):
            # Transition to next phase
            if state.current_phase == PHASE_JOINT_DISCUSSION:
                # Temporary: skip caucus phase
                #     state.current_phase = PHASE_CAUCUS
                #     state.is_in_caucus = True
                #     state.caucus_party = Party(name="REQUESTING_PARTY")
                # elif state.current_phase == PHASE_CAUCUS:
                state.current_phase = PHASE_NEGOTIATION
                state.is_in_caucus = False
                state.caucus_party = None
            elif state.current_phase == PHASE_NEGOTIATION:
                state.current_phase = PHASE_CONCLUSION
            elif state.current_phase == PHASE_CONCLUSION:
                state.current_phase = PHASE_ENDED
                return state

            state.turns_in_current_phase = 0
            state.next_speaker_candidate = Party(
                name="MEDIATOR"
            )  # Mediator typically starts new phases
            return state

        # Get the next speaker from the LLM
        next_speaker = await generate_clerk_decision(llm, state)

        state.next_speaker_candidate = Party(name=next_speaker)

        # Increment counters
        state.turns_in_current_phase += 1
        state.turn_number += 1

        return state

    async def mediator_node(state: MediationState):
        """The mediator node generates the mediator's response"""
        logger.info("⚖️ [MEDIATOR]: Mediator node called")
        state.last_utterance_speaker = Party(name="MEDIATOR")

        if (
            state.current_phase == PHASE_OPENING
            and not state.mediator_opening_statement
        ):
            # Generate opening statement using the new function
            content = await generate_mediator_opening_statement(llm, state)
            state.mediator_opening_statement = content
            summary = await generate_summary(
                llm, content, "This is a mediator's opening statement."
            )
        elif state.current_phase == PHASE_JOINT_DISCUSSION:
            # Generate response using the new function
            content = await generate_mediator_joint_discussion(llm, state)
            summary = await generate_summary(
                llm, content, "This is a mediator's response during joint discussion."
            )
        elif state.current_phase == PHASE_NEGOTIATION:
            content = await generate_negotiation_mediator(llm, state)
            summary = await generate_summary(
                llm, content, "This is a mediator's response during negotiation."
            )
        elif state.current_phase == PHASE_CONCLUSION:
            content = await generate_mediator_conclusion(llm, state)
            summary = await generate_summary(
                llm, content, "This is a mediator's response during conclusion."
            )
        else:
            content = f"Mediator speaking on turn {state.turn_number}."
            summary = await generate_summary(
                llm, content, "This is a mediator's default response."
            )

        # Update state and create event
        state.last_utterance_content = content

        event = MediationEvent(
            mediation_phase=state.current_phase,
            speaker=state.last_utterance_speaker,
            content=content,
            summary=summary,
            token_count=len(content.split()),  # Rough estimate of tokens
        )
        state.events.append(event)

        return state

    async def requesting_party_node(state: MediationState):
        """The requesting party node generates responses"""
        logger.info("🌝 Requesting party node called")
        state.last_utterance_speaker = Party(name="REQUESTING_PARTY")

        if (
            state.current_phase == PHASE_OPENING
            and not state.requesting_party_opening_statement
        ):
            # Get case information using the case query agent
            # content = await case_query_agent.acall_invoke(
            #     query="What was the result of the independent inspection?"
            # )
            content = await generate_requesting_opening_statement(llm, state)
            state.requesting_party_opening_statement = content
            summary = await generate_summary(
                llm, content, "This is a requesting party's opening statement."
            )
        elif state.current_phase == PHASE_JOINT_DISCUSSION:
            # Generate response using the new function
            content = await generate_requesting_joint_discussion(llm, state)
            summary = await generate_summary(
                llm,
                content,
                "This is a requesting party's response during joint discussion.",
            )
        elif state.current_phase == PHASE_NEGOTIATION:
            content = await generate_negotiation_requesting_party(llm, state)
            summary = await generate_summary(
                llm,
                content,
                "This is a requesting party's response during negotiation.",
            )
        elif state.current_phase == PHASE_CONCLUSION:
            content = await generate_conclusion_requesting_party(llm, state)
            summary = await generate_summary(
                llm, content, "This is a requesting party's response during conclusion."
            )
        else:
            content = f"Requesting party speaking on turn {state.turn_number}."
            summary = await generate_summary(
                llm, content, "This is a requesting party's default response."
            )

        # Update state and create event
        state.last_utterance_content = content

        event = MediationEvent(
            mediation_phase=state.current_phase,
            speaker=state.last_utterance_speaker,
            content=content,
            summary=summary,
            token_count=len(content.split()),  # Rough estimate of tokens
        )
        state.events.append(event)

        return state

    async def responding_party_node(state: MediationState):
        """The responding party node generates responses"""
        logger.info("🌚 Responding party node called")
        state.last_utterance_speaker = Party(name="RESPONDING_PARTY")

        if (
            state.current_phase == PHASE_OPENING
            and not state.responding_party_opening_statement
        ):
            # Generate opening statement using the new function
            content = await generate_responding_opening_statement(llm, state)
            state.responding_party_opening_statement = content
            summary = await generate_summary(
                llm, content, "This is a responding party's opening statement."
            )
        elif state.current_phase == PHASE_JOINT_DISCUSSION:
            # Generate response using the new function
            content = await generate_responding_joint_discussion(llm, state)
            summary = await generate_summary(
                llm,
                content,
                "This is a responding party's response during joint discussion.",
            )
        elif state.current_phase == PHASE_NEGOTIATION:
            content = await generate_negotiation_responding_party(llm, state)
            summary = await generate_summary(
                llm,
                content,
                "This is a responding party's response during negotiation.",
            )
        elif state.current_phase == PHASE_CONCLUSION:
            content = await generate_conclusion_responding_party(llm, state)
            summary = await generate_summary(
                llm,
                content,
                "This is a responding party's response during conclusion.",
            )
        else:
            content = f"Responding party speaking on turn {state.turn_number}."
            summary = await generate_summary(
                llm, content, "This is a responding party's default response."
            )

        # Update state and create event
        state.last_utterance_content = content

        event = MediationEvent(
            mediation_phase=state.current_phase,
            speaker=state.last_utterance_speaker,
            content=content,
            summary=summary,
            token_count=len(content.split()),  # Rough estimate of tokens
        )
        state.events.append(event)

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
        lambda x: (
            "end"
            if x.current_phase == PHASE_ENDED
            else {
                "MEDIATOR": "mediator",
                "REQUESTING_PARTY": "requesting_party",
                "RESPONDING_PARTY": "responding_party",
            }[x.next_speaker_candidate.name]
        ),
    )

    app = workflow.compile()

    # Save workflow visualization
    save_workflow_visualization(app)

    async def _response_fn(input_message: str = None) -> str:
        logger.debug("🟢 Starting mediation workflow execution")

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
            turns_in_current_phase=0,  # Initialize phase turn counter
        )

        # Run the workflow
        output = await app.ainvoke(initial_state, {"recursion_limit": 200})

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
