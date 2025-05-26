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
    from .types import Party, MediationPhase
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

    logger.info("🧠 Getting redis memory client")
    memory = builder.get_memory_client("redis_memory")

    logger.info("🤖 Getting LLM with name: %s", config.llm)
    llm = await builder.get_llm(
        llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    logger.info("✅ LLM initialized: %s", llm)

    # Get the case query agent
    # case_query_agent = builder.get_function("case_query_agent")


    # Define phase instances
    PHASE_OPENING = MediationPhase(name="OPENING_STATEMENTS")
    PHASE_JOINT_DISCUSSION = MediationPhase(name="JOINT_DISCUSSION_INFO_GATHERING")
    PHASE_CAUCUS = MediationPhase(name="CAUCUSES")
    PHASE_NEGOTIATION = MediationPhase(name="NEGOTIATION_BARGAINING")
    PHASE_CONCLUSION = MediationPhase(name="CONCLUSION_CLOSING_STATEMENTS")
    PHASE_ENDED = MediationPhase(name="ENDED")

    # --- Pydantic Models for State ---

    class MediationEvent(BaseModel):
        """
        TODO: remove this in favor of using the messages field
        """
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
        session_id: str = ""
        case_title: str = ""
        case_summary: str = ""

        # user role - used in interactive mode to determine the role of the user (requesting or responding party)
        user_role: Optional[Literal["REQUESTING_PARTY", "RESPONDING_PARTY"]] = None

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

        messages: List[BaseMessage] = Field(default_factory=list)

        turn_number: int = 0

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
        state.requesting_party_representative = case_data.get(
            "requesting_party_representative", ""
        )
        state.responding_party_company = case_data.get("responding_party_company", "")
        state.responding_party_representative = case_data.get(
            "responding_party_representative", ""
        )

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

        # Handle conclusion phase
        if state.current_phase == PHASE_CONCLUSION:
            # Strict sequence: requesting party -> responding party -> mediator -> end
            if not state.requesting_party_conclusion:
                state.next_speaker_candidate = Party(name="REQUESTING_PARTY")
            elif not state.responding_party_conclusion:
                state.next_speaker_candidate = Party(name="RESPONDING_PARTY")
            elif not state.mediator_conclusion_settlement:
                state.next_speaker_candidate = Party(name="MEDIATOR")
            else:
                # All conclusions are complete, end the mediation
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
                state.current_phase = PHASE_NEGOTIATION
                state.is_in_caucus = False
                state.caucus_party = None
            elif state.current_phase == PHASE_NEGOTIATION:
                state.next_speaker_candidate = Party(name="MEDIATOR")
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
            state.mediator_conclusion_settlement = content
            summary = await generate_summary(
                llm, content, "This is a mediator's response during conclusion."
            )
        else:
            content = f"Mediator speaking on turn {state.turn_number}."
            summary = await generate_summary(
                llm, content, "This is a mediator's default response."
            )

        # strip the content of any whitespace (qwen3 leaves \n\n after </think>)
        content = content.strip()

        # Update state and create event
        state.last_utterance_content = content

        # add the LLM response message to redis memory
        await memory.add_messages(
            [
                HumanMessage(
                    content=content,
                    additional_kwargs={
                        "speaker": "MEDIATOR",
                        "is_user": False,
                        "phase": state.current_phase.name,
                        "summary": summary,
                    },
                )
            ],
            session_id=state.session_id,
        )

        # update the messages in the state
        state.messages = await memory.get_messages(state.session_id)

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
            state.requesting_party_conclusion = content
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

        # add the LLM response message to redis memory
        await memory.add_messages(
            [
                HumanMessage(
                    content=content,
                    additional_kwargs={
                        "speaker": "REQUESTING_PARTY",
                        "is_user": False,
                        "phase": state.current_phase.name,
                        "summary": summary,
                    },
                )
            ],
            session_id=state.session_id,
        )

        # update the messages in the state
        state.messages = await memory.get_messages(state.session_id)

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
            state.responding_party_conclusion = content
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

        await memory.add_messages(
            [
                HumanMessage(
                    content=content,
                    additional_kwargs={
                        "speaker": "RESPONDING_PARTY",
                        "is_user": False,
                        "phase": state.current_phase.name,
                        "summary": summary,
                    },
                )
            ],
            session_id=state.session_id,
        )

        # update the messages in the state
        state.messages = await memory.get_messages(state.session_id)

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
            # route to END if the next speaker is the user
            or (x.user_role == "REQUESTING_PARTY" and x.next_speaker_candidate.name == "REQUESTING_PARTY")
            or (x.user_role == "RESPONDING_PARTY" and x.next_speaker_candidate.name == "RESPONDING_PARTY")
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

        logger.info("🔍 Input message: %s", input_message)

        from aiq.builder.context import AIQContext
        from starlette.datastructures import QueryParams

        aiq_context = AIQContext.get()
        query_params: QueryParams | None = aiq_context.metadata.query_params
        path_params: dict[str, str] | None = aiq_context.metadata.path_params
        method: str | None = aiq_context.metadata.method

        case_id = path_params.get("case_id") if path_params else input_message

        if not case_id:
            logger.error(
                "🔴 Case ID is required, please provide case ID via --input when using the mediation workflow"
            )
            raise ValueError(
                "Case ID is required, please provide case ID via --input when using the mediation workflow"
            )
        logger.info(f"🔍 Case ID: {case_id}")

        # if using the workflow via CLI, then we start with a fresh state
        if not method:
            session_id = str(uuid.uuid4())[:8]
            initial_state = MediationState(
                case_id=case_id,
                session_id=session_id,
                user_role=None,
                messages=[],
                case_summary="",
                current_phase=PHASE_OPENING,
                turn_number=0,
                turns_in_current_phase=0,
            )
        elif method == "POST":

            # fetch the state from redis memory
            case_data = await memory.get_case_state(case_id)

            # fetch the session id from the query params
            session_id = path_params.get("session_id")

            # there needs to be a session id for the mediation workflow be able to save values
            if not session_id:
                logger.error(
                    "🔴 Session ID is required, please provide session ID via --query when using the mediation workflow"
                )
                raise ValueError(
                    "Session ID is required, please provide session ID via --query when using the mediation workflow"
                )
            logger.info(f"🔍 Session ID: {session_id}")

            role = query_params.get("role", "REQUESTING_PARTY")

            # try to get the current session phase from redis memory, otherwise initialize redis values
            current_phase = await memory.get_session_field(session_id, "current_phase")
            if not current_phase:
                current_phase = PHASE_OPENING.name
                await memory.set_session_field(session_id, "current_phase", current_phase)
            else:
                # Decode bytes to string if needed
                current_phase = current_phase.decode() if isinstance(current_phase, bytes) else current_phase

            current_phase = MediationPhase(name=current_phase)

            # the input_message will be empty on the first request if there are no messages in redis memory
            # if the input_message is not empty, then we add the user message to redis memory
            if input_message != "":
                # add the user message to redis memory
                await memory.add_messages(
                    [
                        HumanMessage(
                            content=input_message,
                            additional_kwargs={
                                "speaker": role,
                                "is_user": True,
                                "phase": current_phase.name,
                            },
                        )
                    ],
                    session_id=session_id,
                )

            # Set the case summary from the basic case information
            case_summary = case_data.get("basic_case_information", "")
            case_title = case_data.get("case_title", "")

            # Set the company names
            requesting_party_company = case_data.get("requesting_party_company", "")
            requesting_party_representative = case_data.get(
                "requesting_party_representative", ""
            )
            responding_party_company = case_data.get("responding_party_company", "")
            responding_party_representative = case_data.get(
                "responding_party_representative", ""
            )

            current_phase = case_data.get("current_phase", PHASE_OPENING.name)
            CURRENT_PHASE = MediationPhase(name=current_phase)

            messages = await memory.get_messages(session_id)

            initial_state = MediationState(
                case_id=case_id,
                session_id=session_id,
                user_role=role,
                case_summary=case_summary,
                case_title=case_title,
                requesting_party_company=requesting_party_company,
                requesting_party_representative=requesting_party_representative,
                responding_party_company=responding_party_company,
                responding_party_representative=responding_party_representative,
                messages=messages,
                current_phase=CURRENT_PHASE,
                turn_number=0,
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
