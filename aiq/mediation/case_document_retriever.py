from typing import Optional
from pydantic import BaseModel, Field
from aiq.builder.function import Function, FunctionInfo
from aiq.builder.builder import Builder
from aiq.data_models.function import FunctionBaseConfig
from aiq.retriever.milvus.register import MilvusRetrieverConfig
from aiq.cli.register_workflow import register_function

# Configuration for the tool
class CaseDocumentRetrieverConfig(FunctionBaseConfig, name="case_document_retriever"):
    """
    Configuration for a tool that retrieves case-specific documents from Milvus.
    """
    retriever: str = Field(
        description="The name of the retriever instance from the workflow configuration"
    )
    collection_name: str = Field(
        default="aiq_case_documents",
        description="The name of the Milvus collection to search"
    )
    top_k: int = Field(
        default=5,
        description="Number of most relevant documents to return"
    )

# Input schema for the tool
class CaseDocumentInputSchema(BaseModel):
    case_id: str = Field(description="The ID of the case to filter documents by")
    question: str = Field(description="The question to search for relevant documents")

# The actual tool implementation
@register_function(config_type=CaseDocumentRetrieverConfig)
async def case_document_retriever(config: CaseDocumentRetrieverConfig, builder: Builder):
    """
    A tool that retrieves case-specific documents from Milvus based on a question and case ID.
    """
    # Get the retriever instance
    retriever = await builder.get_retriever(config.retriever)

    async def _retrieve_case_documents(case_id: str, question: str) -> list[dict]:
        try:
            # Create a filter for the case_id
            filter_expr = f'case_id == "{case_id}"'

            # Perform the search with the filter
            results = await retriever.search(
                query=question,
                collection_name=config.collection_name,
                top_k=config.top_k,
                filters=filter_expr
            )

            # Convert results to a list of dictionaries
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("distance", 0)
                }
                for doc in results.results
            ]

        except Exception as e:
            logger.error(f"Error retrieving case documents: {str(e)}")
            return []

    # Return the function info
    yield FunctionInfo.from_fn(
        fn=_retrieve_case_documents,
        input_schema=CaseDocumentInputSchema,
        description="Retrieves case-specific documents from the vector store based on a question and case ID"
    )