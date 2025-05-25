"""
This file defines the case_generation workflow.

It creates a description of a case for use in a mediation simulation

"""

import logging


class WarningFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.levelname == "WARNING"
            and record.name == "aiq.data_models.discovery_metadata"
            and "Package metadata not found" in record.getMessage()
        )


# Configure logging to suppress warnings for specific modules
root_logger = logging.getLogger()
root_logger.addFilter(WarningFilter())

logging.getLogger("aiq.cli.commands.start").setLevel(logging.ERROR)
logging.getLogger("aiq.data_models.discovery_metadata").setLevel(logging.ERROR)
logging.getLogger("aiq.data_models.discovery_metadata").propagate = False

import os
import random
import string
import json
from pathlib import Path
from typing import TypedDict, List
from pydantic import BaseModel, Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class CaseGenerationWorkflowConfig(FunctionBaseConfig, name="case_generation"):
    # Add your custom configuration parameters here
    llm: LLMRef = "nim_llm"
    data_dir: str = "./data"


class Document(BaseModel):
    """A document in the case"""

    name: str = Field(description="The name of the document")
    description: str = Field(description="A description of what the document contains")
    type: str = Field(description="The type of document (e.g. contract, email, report)")
    filename: str = Field(description="The filename of the document ending with .md")


class DocumentList(BaseModel):
    """List of documents in the case"""

    documents: List[Document] = Field(
        description="List of documents mentioned in the case"
    )


class CaseDetails(BaseModel):
    """Structured case details extracted from the case description"""

    case_title: str = Field(description="The title of the case")
    requesting_party_company: str = Field(
        description="The name of the requesting party's company"
    )
    requesting_party_representative: str = Field(
        description="The name of the requesting party's representative"
    )
    responding_party_company: str = Field(
        description="The name of the responding party's company"
    )
    responding_party_representative: str = Field(
        description="The name of the responding party's representative"
    )


@register_function(
    config_type=CaseGenerationWorkflowConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def case_generation_workflow(
    config: CaseGenerationWorkflowConfig, builder: Builder
):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.output_parsers import PydanticOutputParser
    from langgraph.graph import StateGraph, END

    logger.info("🤖 Getting LLM with name: %s", config.llm)
    llm = await builder.get_llm(
        llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    logger.info("✅ LLM initialized: %s", llm)

    memory = builder.get_memory_client("redis_memory")

    class CaseGenerationState(TypedDict):
        """ "
        Case generation state
        """

        case_id: str | None
        initial_case_description: str | None
        basic_case_information: str | None
        documents: List[dict] | None
        case_title: str = Field(description="The title of the case")
        requesting_party_company: str = Field(
            description="The name of the requesting party's company"
        )
        requesting_party_representative: str = Field(
            description="The name of the requesting party's representative"
        )
        responding_party_company: str = Field(
            description="The name of the responding party's company"
        )
        responding_party_representative: str = Field(
            description="The name of the responding party's representative"
        )

    async def initial(state: CaseGenerationState) -> CaseGenerationState:
        """generate the initial case description"""
        # If case_id is provided, try to load existing case description
        if state.get("case_id"):
            case_dir = Path(config.data_dir) / state["case_id"]
            case_description_file = case_dir / "initial_case_description.md"

            if case_description_file.exists():
                logger.info(
                    f"Loading existing case description from {case_description_file}"
                )
                with open(case_description_file, "r") as f:
                    return {
                        "case_id": state["case_id"],
                        "initial_case_description": f.read(),
                        "documents": None,
                    }

        # Read the prompt from file
        prompt_path = Path(__file__).parent / "prompts" / "initial_case_generation.txt"
        with open(prompt_path, "r") as f:
            case_generation_prompt = f.read().strip()

        # If no existing case found, generate new one
        messages = [
            SystemMessage(
                content="You are a legal case creator that creates descriptions of cases for legal mediation competitions. Your task is to generate information based on the user's request. Your answers should be complete and should address each topic with proper headings."
            ),
            HumanMessage(content=case_generation_prompt),
        ]

        # Get response from LLM
        logger.info("💬 Sending request to LLM")
        response = await llm.ainvoke(messages)

        # Ensure we have content from the response
        content = response.content if hasattr(response, "content") else str(response)

        # save the case description in memory
        await memory.save_case_description(content, state["case_id"])

        # Update the state with the case description
        return {
            "case_id": state.get("case_id"),
            "initial_case_description": content,
            "documents": None,
        }

    async def document_extraction(state: CaseGenerationState) -> CaseGenerationState:
        """Extract structured document information from the case description"""
        # If case_id is provided, try to load existing documents
        if state.get("case_id"):
            case_dir = Path(config.data_dir) / state["case_id"]
            documents_file = case_dir / "documents.json"

            if documents_file.exists():
                logger.info(f"Loading existing documents from {documents_file}")
                with open(documents_file, "r") as f:
                    return {
                        "case_id": state["case_id"],
                        "initial_case_description": state.get(
                            "initial_case_description"
                        ),
                        "documents": json.load(f),
                    }

        if not state.get("initial_case_description"):
            logger.warning("No initial case description found in state")
            return {
                "case_id": state.get("case_id"),
                "initial_case_description": state.get("initial_case_description"),
                "documents": [],
            }

        # Create the output parser
        parser = PydanticOutputParser(pydantic_object=DocumentList)

        # Create messages for the LLM
        messages = [
            SystemMessage(
                content="""You are a document extraction specialist. Your task is to extract information about documents mentioned in the case description.
            For each document mentioned, extract its name, type, and description.
            The output should be a JSON object with a 'documents' key containing an array of document objects.
            Each document object should have 'name', 'type', 'description' and 'filename' fields.
            The filename field should be unique and generated based on the document name with a .md extension.
            ONLY include a valid JSON structure that contains the documents that are explicitly mentioned in the text.
            The first character of your response must be '{', and the last character of your response must be '}'. Do not include any other text, markdown formatting, or explanations.

            Example format:


            {
              "documents": [
                {
                  "name": "Contract Agreement",
                  "type": "contract",
                  "description": "The original contract between parties",
                  "filename": "contract.md"
                }
              ]
            }"""
            ),
            HumanMessage(
                content=f"""Extract document information from this case description:
            {state['initial_case_description']}

            {parser.get_format_instructions()}"""
            ),
        ]

        # Get response from LLM
        logger.info("📑 Sending document extraction request to LLM")
        response = await llm.ainvoke(messages)

        try:
            try:
                raw_json = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning("⚠️ LLM output is not valid JSON:\n%s", response.content)
                return {
                    "case_id": state.get("case_id"),
                    "initial_case_description": state.get("initial_case_description"),
                    "documents": [],
                }

            # Parse the response into structured data
            parsed_output = parser.parse(json.dumps(raw_json))
            documents = [doc.dict() for doc in parsed_output.documents]
            logger.info("✅ Successfully extracted %d documents", len(documents))
            return {
                "case_id": state.get("case_id"),
                "initial_case_description": state.get("initial_case_description"),
                "documents": documents,
            }
        except Exception as e:
            logger.error("❌ Failed to parse document extraction response: %s", str(e))
            return {
                "case_id": state.get("case_id"),
                "initial_case_description": state.get("initial_case_description"),
                "documents": [],
            }

    async def document_generation(state: CaseGenerationState) -> CaseGenerationState:
        """Generate the actual document content for each document in the state"""
        if not state.get("documents"):
            logger.warning("No documents found in state")
            return state

        case_dir = Path(config.data_dir) / state["case_id"]
        documents_dir = case_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        for doc in state["documents"]:
            doc_path = documents_dir / doc["filename"]

            # Skip if document already exists
            if doc_path.exists():
                logger.info(f"Document {doc['filename']} already exists, skipping")
                continue

            # Create messages for the LLM
            messages = [
                SystemMessage(
                    content="""You are an expert document creator specializing in legal competition materials.
                Your task is to create detailed, well-structured documents using proper markdown formatting.
                Use appropriate headings (H1, H2, H3), lists, tables, and other markdown elements to organize the content.
                The document should be professional, clear, and suitable for a legal competition.
                Include all relevant details and maintain a formal tone throughout."""
                ),
                HumanMessage(
                    content=f"""Create a detailed document for a legal competition with the following information:

                Case Information:
                {state.get('basic_case_information', '')}

                Information about the document to create:
                Document Name: {doc['name']}
                Document Type: {doc['type']}
                Document Description: {doc['description']}

                Please create a comprehensive document based on the provided document information.
                The document should be detailed enough to be used in a legal competition."""
                ),
            ]

            # Get response from LLM
            logger.info(f"📄 Generating document content for {doc['filename']}")
            response = await llm.ainvoke(messages)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Save the document
            with open(doc_path, "w") as f:
                f.write(content)
            logger.info(f"✅ Saved document to {doc_path}")

        return state

    async def basic_case_information_extraction(
        state: CaseGenerationState,
    ) -> CaseGenerationState:
        """Extract basic case information from the case description"""
        if not state.get("initial_case_description"):
            logger.warning("No initial case description found in state")
            return {
                "case_id": state.get("case_id"),
                "initial_case_description": state.get("initial_case_description"),
                "basic_case_information": None,
                "documents": state.get("documents"),
            }

        # Create messages for the LLM
        messages = [
            SystemMessage(
                content="""You are a legal case information extraction specialist. Your task is to extract basic, non-confidential information from case descriptions.
            Focus on extracting:
            1. A clear, concise case title
            2. All parties involved in the case, including full names of companies and representatives
            3. Background information about the case
            4. General facts that are known to all parties

            DO NOT include any confidential information or facts that are specific to individual parties.
            The information should be structured and clear, suitable for use in legal proceedings.
            Keep the description of each section concise and to the point."""
            ),
            HumanMessage(
                content=f"""Extract basic case information from this case description:
            {state['initial_case_description']}

            Format your response with clear headings for each section:
            # Case Title
            [Title]

            # Parties Involved
            [List of parties]

            # Background
            [Background information]

            # General Facts
            [General facts known to all parties]"""
            ),
        ]

        # Get response from LLM
        logger.info("📝 Doing basic case information extraction")
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Save the basic case information to a markdown file
        case_dir = Path(config.data_dir) / state["case_id"]
        case_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create directory if it doesn't exist
        basic_info_file = case_dir / "basic_case_information.md"
        with open(basic_info_file, "w") as f:
            f.write(content)
        logger.info(f"✅ Saved basic case information to {basic_info_file}")

        return {
            "case_id": state.get("case_id"),
            "initial_case_description": state.get("initial_case_description"),
            "basic_case_information": content,
            "documents": state.get("documents"),
        }

    async def case_details_extraction(
        state: CaseGenerationState,
    ) -> CaseGenerationState:
        """Extract structured case details from the basic case information"""
        if not state.get("basic_case_information"):
            logger.warning("No basic case information found in state")
            return state

        # Create the output parser
        parser = PydanticOutputParser(pydantic_object=CaseDetails)

        # Create messages for the LLM
        messages = [
            SystemMessage(
                content="""You are a case details extraction specialist. Your task is to extract specific case details from the provided case information.
            Extract the following information:
                - case title
                - requesting party company name
                - requesting party representative name (the name of the individual at the company who is representing the requesting party)
                - responding party company name
                - responding party representative name (the name of the individual at the company who is representing the responding party)

            The first character of your response must be '{', and the last character of your response must be '}'.
            Do not include any other text, markdown formatting, or explanations.
            Only return valid JSON that matches the required schema."""
            ),
            HumanMessage(
                content=f"""Extract case details from this case information:
            {state['initial_case_description']}

            {parser.get_format_instructions()}"""
            ),
        ]

        # Get response from LLM
        logger.info("📋 Extracting case details")
        response = await llm.ainvoke(messages)
        try:
            # Parse the response into structured data
            parsed_output = parser.parse(response.content.strip())
            case_details = parsed_output.dict()

            # Save the case details to a JSON file
            case_dir = Path(config.data_dir) / state["case_id"]
            case_details_file = case_dir / "case_details.json"
            with open(case_details_file, "w") as f:
                json.dump(case_details, f, indent=2)
            logger.info(f"✅ Saved case details to {case_details_file}")

            ret_state = {**state, **case_details}

            return ret_state
        except Exception as e:
            logger.error("❌ Failed to parse case details: %s", str(e))
            return state

    async def case_image_generation_prompts(
        state: CaseGenerationState,
    ) -> CaseGenerationState:
        """Generate image prompts for the case"""
        if not state.get("basic_case_information"):
            logger.warning("No basic case information found in state")
            return state

        # Create messages for the LLM
        messages = [
            SystemMessage(
                content="""You are an expert at creating detailed image generation prompts for legal cases. Your task is to generate a JSON array of image prompts that can be used with AI image generation services.

            Each prompt should be concise but detailed, capturing different aspects of the case such as:
            - The physical appearance and attire of the parties involved
            - The business environment and facilities
            - The industry context and setting
            - Key locations or scenes relevant to the case
            - Professional settings like courtrooms or meeting rooms
            - Any relevant documents or evidence

            The prompts should be specific enough to generate meaningful images but concise enough to work well with image generation services.
            Focus on creating diverse prompts that capture different aspects of the case.

            The first character of your response must be '[', and the last character of your response must be ']'.
            Do not include any other text, markdown formatting, or explanations.
            Only return a valid JSON array of strings.

            Example format:
            [
                "A tense boardroom meeting with executives in tailored suits, one group seated at a long mahogany table reviewing legal documents, while others stand presenting charts on a large LED display, natural light streaming through floor-to-ceiling windows",
                "A high-tech manufacturing facility with automated assembly lines, workers in safety gear monitoring quality control stations, and digital displays showing production metrics, capturing the industrial scale of operations",
                "A modern courtroom with a judge in black robes presiding from an elevated bench, lawyers in formal attire presenting evidence on large screens, and a diverse jury panel seated in the gallery, all under dramatic lighting",
                "A corporate headquarters lobby featuring a grand staircase, reception desk with security personnel, and digital displays showing company achievements, with professionals in business attire entering and exiting",
                "A secure document storage room with rows of filing cabinets, a large digital archive system, and a team of paralegals carefully organizing and digitizing case materials under bright task lighting",
                "A mediation room with a circular table, neutral decor, and comfortable seating for all parties, featuring a professional mediator facilitating discussion while participants review settlement documents",
                "An outdoor construction site with heavy machinery, safety barriers, and workers in hard hats inspecting blueprints, showing the physical location central to the dispute",
                "A sophisticated data center with server racks, monitoring stations, and IT professionals analyzing network security logs, representing the technological aspects of the case"
            ]"""
            ),
            HumanMessage(
                content=f"""Generate image prompts based on this case information:
            {state['basic_case_information']}

            Include prompts that capture the nature of the case, the parties involved, and the business context."""
            ),
        ]

        # Get response from LLM
        logger.info("🎨 Generating image prompts")
        response = await llm.ainvoke(messages)
        try:
            # Parse the response into a list of prompts
            prompts = json.loads(response.content.strip())

            # Save the prompts to a JSON file
            case_dir = Path(config.data_dir) / state["case_id"]
            prompts_file = case_dir / "image_prompts.json"
            with open(prompts_file, "w", encoding='utf-8') as f:
                json.dump(prompts, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved image prompts to {prompts_file}")

            return {**state, "image_prompts": prompts}
        except Exception as e:
            logger.error("❌ Failed to parse image prompts: %s", str(e))
            return state

    workflow = StateGraph(CaseGenerationState)
    workflow.add_node("initial", initial)
    workflow.set_entry_point("initial")
    workflow.add_node("document_extraction", document_extraction)
    workflow.add_node(
        "basic_case_information_extraction", basic_case_information_extraction
    )
    workflow.add_node("document_generation", document_generation)
    workflow.add_node("case_details_extraction", case_details_extraction)
    workflow.add_node("case_image_generation_prompts", case_image_generation_prompts)

    # Update the edges to create a proper flow
    workflow.add_edge("initial", "document_extraction")
    workflow.add_edge("document_extraction", "basic_case_information_extraction")
    workflow.add_edge("basic_case_information_extraction", "document_generation")
    workflow.add_edge("document_generation", "case_details_extraction")
    workflow.add_edge("case_details_extraction", "case_image_generation_prompts")
    workflow.add_edge("case_image_generation_prompts", END)

    app = workflow.compile()

    # Save workflow visualization
    try:
        import graphviz

        dot = graphviz.Digraph(comment="Case Generation Workflow")
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="#4A90E2",
            fontcolor="white",
            fontname="Arial",
        )
        dot.attr("edge", color="#666666", penwidth="1.5")

        # Define node colors for different types of nodes
        node_colors = {
            "initial": "#4A90E2",  # Blue
            "basic_case_information_extraction": "#50C878",  # Emerald Green
            "document_extraction": "#FFA500",  # Orange
            "document_generation": "#9370DB",  # Medium Purple
            "END": "#FF6B6B",  # Coral Red
        }

        # Add nodes with custom colors
        for node in app.get_graph().nodes:
            color = node_colors.get(
                node, "#4A90E2"
            )  # Default to blue if node type not specified
            dot.node(node, node, fillcolor=color)

        # Add edges
        for edge in app.get_graph().edges:
            dot.edge(edge[0], edge[1])

        # Save the graph
        dot.render("case_generation_workflow", format="png", cleanup=True)
        logger.info("📊 Saved workflow visualization to case_generation_workflow.png")
    except ImportError:
        logger.warning(
            "⚠️ graphviz not installed. Skipping workflow visualization. Install with: pip install graphviz"
        )
    except Exception as e:
        logger.error(f"❌ Failed to save workflow visualization: {str(e)}")

    async def _response_fn(input_message: str = None) -> str:
        logger.debug("🚀 Starting case_generation workflow execution")

        case_id = input_message or "".join(random.choices(string.ascii_lowercase, k=8))
        # Initialize the state with the required fields
        initial_state = {
            "case_id": case_id,  # Use input_message as case_id
            "initial_case_description": None,
            "documents": None,
        }

        out = await app.ainvoke(initial_state)
        output = out["initial_case_description"]
        documents = out.get("documents", [])

        # Create the directory structure
        case_dir = Path(config.data_dir) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save the case description to a markdown file
        output_file = case_dir / "initial_case_description.md"
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(output)

        # Save the documents to a JSON file
        documents_file = case_dir / "documents.json"
        with open(documents_file, "w", encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved case description to {output_file}")
        logger.info(f"💾 Saved documents to {documents_file}")

        # Write the CaseGenerationState to a YAML file
        import yaml

        # Add custom representer for strings
        def str_presenter(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_presenter)

        state_file = case_dir / "case_generation_state.yaml"

        # Create a clean state dictionary for YAML serialization
        state_dict = {
            "case_id": out.get("case_id"),
            "initial_case_description": out.get("initial_case_description"),
            "basic_case_information": out.get("basic_case_information"),
            "case_title": out.get("case_title"),
            "requesting_party_company": out.get("requesting_party_company"),
            "requesting_party_representative": out.get(
                "requesting_party_representative"
            ),
            "responding_party_company": out.get("responding_party_company"),
            "responding_party_representative": out.get(
                "responding_party_representative"
            ),
            "documents": [
                {
                    "name": doc.get("name"),
                    "type": doc.get("type"),
                    "description": doc.get("description"),
                    "filename": doc.get("filename"),
                }
                for doc in (out.get("documents") or [])
            ],
        }

        logger.info(f"🧠 Saving case state to memory: {state_dict}")
        await memory.save_case_state(state_dict, case_id)

        with open(state_file, "w", encoding="utf-8") as f:
            yaml.dump(
                state_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,  # Allow Unicode characters
                width=1000,  # Prevent line wrapping
            )

        return out["case_id"]

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("❌ Exited early!", exc_info=True)
    finally:
        logger.debug("🧹 Cleaning up case_generation workflow.")
