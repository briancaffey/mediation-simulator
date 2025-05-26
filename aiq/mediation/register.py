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
    from .types import PartyType, MediationPhaseType
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

    # Define phase constants
    PHASE_OPENING = "OPENING_STATEMENTS"
    PHASE_JOINT_DISCUSSION = "JOINT_DISCUSSION_INFO_GATHERING"
    PHASE_CAUCUS = "CAUCUSES"
    PHASE_NEGOTIATION = "NEGOTIATION_BARGAINING"
    PHASE_CONCLUSION = "CONCLUSION_CLOSING_STATEMENTS"
    PHASE_ENDED = "ENDED"

    # --- Pydantic Models for State ---

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

        current_phase: MediationPhaseType = Field(default=PHASE_OPENING)

        messages: List[BaseMessage] = Field(default_factory=list)

        turn_number: int = 0

        # For clerk processing
        last_utterance_speaker: Optional[PartyType] = None

        # For clerk decision-making
        next_speaker_candidate: Optional[PartyType] = None

        # Configuration for phase transitions
        max_turns_per_phase: Dict[str, int] = Field(
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

        return state

    async def clerk_node(state: MediationState):
        """The clerk node decides who speaks next and whether to end the mediation"""
        logger.info(
            f"👤 [CLERK] Starting clerk node - Phase: {state.current_phase}, Turn: {state.turn_number}"
        )

        # # First check if it's the user's turn to speak
        # if (state.user_role == "REQUESTING_PARTY" and state.next_speaker_candidate == "REQUESTING_PARTY") or \
        #    (state.user_role == "RESPONDING_PARTY" and state.next_speaker_candidate == "RESPONDING_PARTY"):
        #     logger.info(f"👤 [CLERK] It's the user's turn to speak (role: {state.user_role}) - ending workflow")
        #     return state

        # Check if we should end the mediation
        if state.current_phase == PHASE_ENDED:
            logger.info("🏁 [CLERK] Mediation has ended - no more turns needed")
            return state

        # Handle opening phase
        if state.current_phase == PHASE_OPENING:
            logger.info("🎬 [CLERK] In Opening Phase - checking opening statements")
            # Check opening statements in order: mediator, requesting party, responding party
            if not state.mediator_opening_statement:
                logger.info("👨‍⚖️ [CLERK] Mediator needs to give opening statement")
                state.next_speaker_candidate = "MEDIATOR"
            elif not state.requesting_party_opening_statement:
                logger.info("👥 [CLERK] Requesting party needs to give opening statement")
                state.next_speaker_candidate = "REQUESTING_PARTY"
            elif not state.responding_party_opening_statement:
                logger.info("👥 [CLERK] Responding party needs to give opening statement")
                state.next_speaker_candidate = "RESPONDING_PARTY"
            else:
                logger.info("✅ [CLERK] All opening statements complete - transitioning to joint discussion")
                # All opening statements are complete, transition to joint discussion
                state.current_phase = PHASE_JOINT_DISCUSSION
                state.turns_in_current_phase = 0
                state.next_speaker_candidate = "MEDIATOR"  # Mediator typically starts joint discussion
            return state

        # Handle conclusion phase
        if state.current_phase == PHASE_CONCLUSION:
            logger.info("🎯 [CLERK] In Conclusion Phase - checking conclusion statements")
            # Strict sequence: requesting party -> responding party -> mediator -> end
            if not state.requesting_party_conclusion:
                logger.info("👥 [CLERK] Requesting party needs to give conclusion")
                state.next_speaker_candidate = "REQUESTING_PARTY"
            elif not state.responding_party_conclusion:
                logger.info("👥 [CLERK] Responding party needs to give conclusion")
                state.next_speaker_candidate = "RESPONDING_PARTY"
            elif not state.mediator_conclusion_settlement:
                logger.info("👨‍⚖️ [CLERK] Mediator needs to give final settlement")
                state.next_speaker_candidate = "MEDIATOR"
            else:
                logger.info("🏁 [CLERK] All conclusions complete - ending mediation")
                # All conclusions are complete, end the mediation
                state.current_phase = PHASE_ENDED
            return state

        # For all other phases, use the LLM to decide the next speaker
        # Check if we've reached max turns for the current phase
        if (
            state.turns_in_current_phase
            >= state.max_turns_per_phase[state.current_phase]
        ):
            logger.info(f"⏰ [CLERK] Max turns ({state.max_turns_per_phase[state.current_phase]}) reached for current phase")
            # Transition to next phase
            if state.current_phase == PHASE_JOINT_DISCUSSION:
                logger.info("🔄 [CLERK] Transitioning from joint discussion to negotiation")
                state.current_phase = PHASE_NEGOTIATION
            elif state.current_phase == PHASE_NEGOTIATION:
                logger.info("🔄 [CLERK] Transitioning from negotiation to conclusion")
                state.next_speaker_candidate = "MEDIATOR"
                state.current_phase = PHASE_CONCLUSION
            elif state.current_phase == PHASE_CONCLUSION:
                logger.info("🏁 [CLERK] Conclusion phase complete - ending mediation")
                state.current_phase = PHASE_ENDED
                return state

            state.turns_in_current_phase = 0
            state.next_speaker_candidate = "MEDIATOR"  # Mediator typically starts new phases
            logger.info("👨‍⚖️ [CLERK] Mediator will start the new phase")
            return state

        # Get the next speaker from the LLM
        logger.info("🤔 [CLERK] Using LLM to decide next speaker")
        next_speaker = await generate_clerk_decision(llm, state)
        state.next_speaker_candidate = next_speaker
        logger.info(f"🎯 [CLERK] LLM selected next speaker: {next_speaker}")

        # Increment counters
        state.turns_in_current_phase = int(state.turns_in_current_phase) + 1
        state.turn_number = int(state.turn_number) + 1
        logger.info(f"📊 [CLERK] Updated counters - Turn: {state.turn_number}, Phase turns: {state.turns_in_current_phase}")

        return state

    async def mediator_node(state: MediationState):
        """The mediator node generates the mediator's response"""
        logger.info("⚖️ [MEDIATOR]: Mediator node called")
        state.last_utterance_speaker = "MEDIATOR"

        if (
            state.current_phase == PHASE_OPENING
            and not state.mediator_opening_statement
        ):
            # Generate opening statement using the new function
            logger.info("👨‍⚖️ [MEDIATOR] Generating mediator's opening statement")
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

        # add the LLM response message to redis memory
        await memory.add_messages(
            [
                HumanMessage(
                    content=content,
                    additional_kwargs={
                        "speaker": "MEDIATOR",
                        "is_user": False,
                        "phase": state.current_phase,
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
        state.last_utterance_speaker = "REQUESTING_PARTY"

        if (
            state.current_phase == PHASE_OPENING
            and not state.requesting_party_opening_statement
        ):
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

        # add the LLM response message to redis memory
        await memory.add_messages(
            [
                HumanMessage(
                    content=content,
                    additional_kwargs={
                        "speaker": "REQUESTING_PARTY",
                        "is_user": False,
                        "phase": state.current_phase,
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
        state.last_utterance_speaker = "RESPONDING_PARTY"

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

        await memory.add_messages(
            [
                HumanMessage(
                    content=content,
                    additional_kwargs={
                        "speaker": "RESPONDING_PARTY",
                        "is_user": False,
                        "phase": state.current_phase,
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
            or (x.user_role == "REQUESTING_PARTY" and x.next_speaker_candidate == "REQUESTING_PARTY")
            or (x.user_role == "RESPONDING_PARTY" and x.next_speaker_candidate == "RESPONDING_PARTY")
            else {
                "MEDIATOR": "mediator",
                "REQUESTING_PARTY": "requesting_party",
                "RESPONDING_PARTY": "responding_party",
            }[x.next_speaker_candidate]
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

        # get the case id from the path params or the input message if using the CLI
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

        # if using the workflow via API, then we load the state from redis memory
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
            user_role = query_params.get("role", "REQUESTING_PARTY")

            # get the mediation_session_state from redis memory
            mediation_session_state_from_memory = await memory.get_session_state(session_id)

            logger.info(f"🔍 Mediation session state from memory:\n\n\n {mediation_session_state_from_memory}\n\n\n")

            mediation_session_state = {}
            mediation_session_state["case_id"] = case_id
            mediation_session_state["session_id"] = session_id
            mediation_session_state["case_summary"] = case_data.get("basic_case_information", "")
            mediation_session_state["user_role"] = user_role
            # names
            mediation_session_state["case_title"] = case_data.get("case_title", "")
            mediation_session_state["requesting_party_company"] = case_data.get("requesting_party_company", "")
            mediation_session_state["requesting_party_representative"] = case_data.get("requesting_party_representative", "")
            mediation_session_state["responding_party_company"] = case_data.get("responding_party_company", "")
            mediation_session_state["responding_party_representative"] = case_data.get("responding_party_representative", "")

            if not mediation_session_state_from_memory:
                mediation_session_state["current_phase"] = PHASE_OPENING
                mediation_session_state["turn_number"] = 0
                mediation_session_state["turns_in_current_phase"] = 0

                # opening statements
                mediation_session_state["mediator_opening_statement"] = ""
                mediation_session_state["requesting_party_opening_statement"] = ""
                mediation_session_state["responding_party_opening_statement"] = ""

                # conclusion statements
                mediation_session_state["mediator_conclusion_settlement"] = ""
                mediation_session_state["requesting_party_conclusion"] = ""
                mediation_session_state["responding_party_conclusion"] = ""
            else:
                # pass
                mediation_session_state["mediator_opening_statement"] = mediation_session_state_from_memory.get("mediator_opening_statement", "")
                mediation_session_state["requesting_party_opening_statement"] = mediation_session_state_from_memory.get("requesting_party_opening_statement", "")
                mediation_session_state["responding_party_opening_statement"] = mediation_session_state_from_memory.get("responding_party_opening_statement", "")
                mediation_session_state["mediator_conclusion_settlement"] = mediation_session_state_from_memory.get("mediator_conclusion_settlement", "")
                mediation_session_state["requesting_party_conclusion"] = mediation_session_state_from_memory.get("requesting_party_conclusion", "")
                mediation_session_state["responding_party_conclusion"] = mediation_session_state_from_memory.get("responding_party_conclusion", "")
                mediation_session_state["current_phase"] = mediation_session_state_from_memory.get("current_phase", PHASE_OPENING)

            # the input_message will be empty on the first request if there are no messages in redis memory
            # if the input_message is not empty, then we add the user message to redis memory
            if input_message != "":
                # add the user message to redis memory
                await memory.add_messages(
                    [
                        HumanMessage(
                            content=input_message,
                            additional_kwargs={
                                "speaker": user_role,
                                "is_user": True,
                                "phase": mediation_session_state["current_phase"],
                                "summary": input_message,
                            },
                        )
                    ],
                    session_id=session_id,
                )

                # if the current phase is opening, then we need to add the opening statements to the state
                if mediation_session_state["current_phase"] == PHASE_OPENING:
                    # if the user is requesting party, then we need to add the requesting party's opening statement to the state
                    if user_role == "REQUESTING_PARTY":
                        mediation_session_state["requesting_party_opening_statement"] = input_message
                    # if the user is responding party, then we need to add the responding party's opening statement to the state
                    else:
                        mediation_session_state["responding_party_opening_statement"] = input_message

                # if the current phase is conclusion, then we need to add the conclusion statements to the state
                if mediation_session_state["current_phase"] == PHASE_CONCLUSION:
                    # if the user is requesting party, then we need to add the requesting party's conclusion statement to the state
                    if user_role == "REQUESTING_PARTY":
                        mediation_session_state["requesting_party_conclusion"] = input_message
                    # if the user is responding party, then we need to add the responding party's conclusion statement to the state
                    else:
                        mediation_session_state["responding_party_conclusion"] = input_message

            messages = await memory.get_messages(session_id)

            initial_state = MediationState(
                **mediation_session_state,
                messages=messages,
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

        # save the mediation session state to redis memory
        # don't save the chat messages since those are stored separately in redis
        del state_dict["messages"]
        await memory.save_session_state(state_dict, session_id)

        return output.get("case_id")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up case_generation workflow.")
