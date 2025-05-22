# aiq/index.py

import os
import json
import logging
import traceback
from pathlib import Path
import sys

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from llama_index.vector_stores.milvus import MilvusVectorStore

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("aiq_indexing.log")],
)
logger = logging.getLogger(__name__)

# Define constants for storage configuration
MILVUS_COLLECTION_NAME = "mediation_simulator_case_documents"


def get_index():
    """
    Gets the index from Milvus server.
    Returns the index object.
    """
    logger.info("Starting index initialization...")

    VECTORDB_SERVICE_HOST = os.environ.get("VECTORDB_SERVICE_HOST", "localhost")
    VECTORDB_SERVICE_PORT = os.environ.get("VECTORDB_SERVICE_PORT", "19530")

    logger.info(
        f"Configuration: VECTORDB_SERVICE_HOST={VECTORDB_SERVICE_HOST}, "
        f"VECTORDB_SERVICE_PORT={VECTORDB_SERVICE_PORT}"
    )

    try:
        logger.info("Initializing embedding model: nvidia/nv-embedqa-e5-v5")
        embed_model = NVIDIAEmbeddings(
            base_url="http://192.168.5.96:8000/v1", model="nvidia/nv-embedqa-e5-v5"
        )
        Settings.embed_model = embed_model
        logger.info("✅ Embedding model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedding model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    VECTORDB_URI = f"http://{VECTORDB_SERVICE_HOST}:{VECTORDB_SERVICE_PORT}"
    logger.info(f"🦅 Connecting to Milvus at {VECTORDB_URI}")

    try:
        vector_store = MilvusVectorStore(
            uri=VECTORDB_URI,
            dim=1024,  # Updated dimension for nv-embedqa-e5-v5
            collection_name=MILVUS_COLLECTION_NAME,
            overwrite=True,  # Set to True to recreate the collection
            batch_size=50,  # Add smaller batch size to prevent row count mismatches
            embedding_field="vector",
            metric_type="IP",  # Set metric type to Inner Product
        )
        logger.info("✅ Successfully created MilvusVectorStore instance")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("✅ Created storage context with Milvus vector store")

        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        logger.info(
            f"✅ Successfully initialized index with Milvus collection '{MILVUS_COLLECTION_NAME}'"
        )
        return index
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def build_index():
    """
    Loops over case_id folders, reads document metadata and content,
    processes them into LlamaIndex nodes with special metadata and prepended description headers,
    and indexes them using Milvus.
    """
    logger.info("🚀 Starting index build process")

    script_dir = Path(__file__).parent
    data_root_dir = script_dir / "data"
    logger.info(f"Data root directory: {data_root_dir}")

    if not data_root_dir.is_dir():
        logger.error(f"❌ Data root directory not found: {data_root_dir}")
        return

    all_llama_documents = []
    processed_cases = 0
    skipped_cases = 0
    processed_docs = 0
    skipped_docs = 0

    for case_id_folder in data_root_dir.iterdir():
        if not case_id_folder.is_dir():
            continue

        case_id = case_id_folder.name
        logger.info(f"📁 Processing case folder: {case_id}")

        if not (len(case_id) == 8 and case_id.isalnum()):
            logger.warning(f"⚠️ Invalid case_id format: {case_id} (skipping)")
            skipped_cases += 1
            continue

        documents_json_path = case_id_folder / "documents.json"
        docs_content_folder = case_id_folder / "documents"

        if not documents_json_path.exists():
            logger.warning(f"⚠️ documents.json not found in {case_id_folder}")
            skipped_cases += 1
            continue
        if not docs_content_folder.is_dir():
            logger.warning(f"⚠️ 'documents' subfolder not found in {case_id_folder}")
            skipped_cases += 1
            continue

        try:
            logger.info(f"📄 Reading documents.json for case {case_id}")
            with open(documents_json_path, "r", encoding="utf-8") as f:
                doc_metadata_list = json.load(f)
            logger.info(
                f"✅ Successfully loaded {len(doc_metadata_list)} document entries"
            )
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decoding JSON from {documents_json_path}: {str(e)}")
            logger.error(traceback.format_exc())
            skipped_cases += 1
            continue
        except Exception as e:
            logger.error(f"❌ Unexpected error reading documents.json: {str(e)}")
            logger.error(traceback.format_exc())
            skipped_cases += 1
            continue

        for doc_info in doc_metadata_list:
            doc_name = doc_info.get("name")
            doc_description = doc_info.get("description")
            doc_type = doc_info.get("type")
            doc_filename = doc_info.get("filename")

            logger.info(f"📑 Processing document: {doc_filename}")

            if not all([doc_name, doc_description, doc_type, doc_filename]):
                logger.warning(
                    f"⚠️ Missing required fields in document metadata: {doc_info}"
                )
                skipped_docs += 1
                continue

            file_path = docs_content_folder / doc_filename
            if not file_path.is_file():
                logger.warning(f"⚠️ Document file not found: {file_path}")
                skipped_docs += 1
                continue

            try:
                logger.info(f"📖 Reading document content from {file_path}")
                with open(file_path, "r", encoding="utf-8") as f_content:
                    markdown_content = f_content.read()
                logger.info(f"✅ Successfully read {len(markdown_content)} characters")
            except Exception as e:
                logger.error(f"❌ Error reading file {file_path}: {str(e)}")
                logger.error(traceback.format_exc())
                skipped_docs += 1
                continue

            try:
                llama_doc = Document(
                    text=markdown_content,
                    metadata={
                        "case_id": case_id,
                        "document_name": doc_name,
                        "document_description": doc_description,
                        "document_type": doc_type,
                        "filename": doc_filename,
                    },
                    metadata_separator="::",
                    metadata_template="{key}=>{value}",
                    text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
                )
                all_llama_documents.append(llama_doc)
                processed_docs += 1
                logger.info(
                    f"✅ Successfully created LlamaIndex Document for: {doc_filename}"
                )
            except Exception as e:
                logger.error(f"❌ Error creating LlamaIndex Document: {str(e)}")
                logger.error(traceback.format_exc())
                skipped_docs += 1
                continue

        processed_cases += 1

    logger.info(f"📊 Processing Summary:")
    logger.info(f"Total cases processed: {processed_cases}")
    logger.info(f"Cases skipped: {skipped_cases}")
    logger.info(f"Documents processed: {processed_docs}")
    logger.info(f"Documents skipped: {skipped_docs}")
    logger.info(f"Total LlamaIndex Documents created: {len(all_llama_documents)}")

    if not all_llama_documents:
        logger.warning("⚠️ No valid documents found to process")
        return

    try:
        logger.info("🔄 Getting Milvus index")
        index = get_index()

        logger.info("🔧 Configuring node parser")
        node_parser = SentenceSplitter(
            chunk_size=150,  # Much smaller chunks to ensure we stay under limit
            chunk_overlap=10,  # Keep small overlap
            separator="\n",  # Split on newlines to maintain better context
        )
        logger.info("✅ Node parser configured")

        all_processed_nodes = []
        for l_doc in all_llama_documents:
            logger.info(f"📝 Processing document: {l_doc.metadata['filename']}")
            try:
                nodes_from_doc = node_parser.get_nodes_from_documents([l_doc])
                logger.info(f"✅ Generated {len(nodes_from_doc)} nodes")

                temp_nodes_for_this_doc = []
                for node in nodes_from_doc:
                    # Keep the document description very short
                    doc_description_for_header = node.metadata.get(
                        "document_description", "No description provided."
                    )[:30]
                    original_node_content = node.get_content(metadata_mode="none")

                    # Truncate the content if it's too long (rough estimate: 1 token ≈ 4 characters)
                    max_chars = 400  # Conservative estimate for 512 tokens
                    if len(original_node_content) > max_chars:
                        original_node_content = (
                            original_node_content[:max_chars] + "..."
                        )

                    # Make the header as short as possible
                    node.text = (
                        f"Doc:{doc_description_for_header}\n{original_node_content}"
                    )

                    # Log the length of the text to help debug
                    logger.info(f"Node text length: {len(node.text)} characters")

                    temp_nodes_for_this_doc.append(node)

                all_processed_nodes.extend(temp_nodes_for_this_doc)
                logger.info(
                    f"✅ Successfully processed {len(temp_nodes_for_this_doc)} nodes"
                )
            except Exception as e:
                logger.error(
                    f"❌ Error processing document {l_doc.metadata['filename']}: {str(e)}"
                )
                logger.error(traceback.format_exc())
                continue

        if not all_processed_nodes:
            logger.warning("⚠️ No nodes were generated from the documents")
            return

        logger.info(f"📥 Inserting {len(all_processed_nodes)} nodes into Milvus")
        index.insert_nodes(all_processed_nodes)
        logger.info("✅ Successfully inserted nodes into Milvus")

    except Exception as e:
        logger.error(f"❌ Error during index building: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    logger.info("\n✨ Index building process completed successfully")


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting AIQ indexing script")
        build_index()
        logger.info("✨ Script completed successfully")
        sys.exit(0)  # Clean exit
    except Exception as e:
        logger.error(f"❌ Fatal error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Exit with error code

    # --- Instructions for running ---
    # 1. Ensure you have a Python environment with necessary packages
    #
    # 2. Set up your data directory structure:
    #    aiq/
    #    ├── index.py  (this script)
    #    └── data/
    #        └── CASEID01/  (e.g., an 8-character case_id)
    #            ├── documents.json
    #            └── documents/
    #                ├── some_document.md
    #                └── another_document.md
    #
    # 3. Configure Milvus:
    #    - Ensure Milvus server is running and accessible
    #    - Set environment variables:
    #      export VECTORDB_SERVICE_HOST="your_milvus_host"  (e.g., "localhost")
    #      export VECTORDB_SERVICE_PORT="19530"
    #
    # 4. Run the script:
    #    (from parent of aiq) python -m aiq.index
    #    (from within aiq)   python index.py
