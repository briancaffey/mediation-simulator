from typing import Optional
from pydantic import BaseModel, Field
from aiq.builder.function import Function, FunctionInfo
from aiq.builder.builder import Builder
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import LLMRef
from aiq.retriever.milvus.register import MilvusRetrieverConfig
from aiq.cli.register_workflow import register_function
from aiq.builder.framework_enum import LLMFrameworkEnum

import logging

logger = logging.getLogger(__name__)


# Configuration for the tool
class CaseDocumentRAGConfig(FunctionBaseConfig, name="case_document_rag"):
    """
    Configuration for a RAG system that retrieves and answers questions about case documents.
    """

    retriever: str = Field(
        description="The name of the retriever instance from the workflow configuration"
    )
    llm_name: LLMRef = Field(
        description="The name of the LLM to use for answering questions"
    )
    collection_name: str = Field(
        default="aiq_case_documents",
        description="The name of the Milvus collection to search",
    )
    top_k: int = Field(
        default=5, description="Number of most relevant documents to return"
    )


# Output schema for the tool
class CaseDocumentOutputSchema(BaseModel):
    answer: str = Field(
        description="The answer to the question based on the retrieved documents"
    )
    documents: list[dict] = Field(
        description="The retrieved documents used to generate the answer"
    )


# The actual tool implementation
@register_function(config_type=CaseDocumentRAGConfig)
async def case_document_rag(config: CaseDocumentRAGConfig, builder: Builder):
    """
    A RAG system that retrieves case-specific documents and uses an LLM to answer questions.
    """
    # Get the retriever instance
    retriever = await builder.get_retriever(config.retriever)

    # Get the LLM instance
    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    async def _answer_case_question(question: str) -> CaseDocumentOutputSchema:
        try:
            # TODO: Add a filter for the case_id
            # Create a filter for the case_id
            # filter_expr = f'case_id == "{case_id}"'

            # Perform the search with the filter
            results = await retriever.search(
                query=question,
                collection_name=config.collection_name,
                top_k=config.top_k,
                # filters=filter_expr
            )

            # Convert results to a list of dictionaries
            documents = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("distance", 0),
                }
                for doc in results.results
            ]

            # Prepare the context for the LLM
            context = "\n\n".join([doc["content"] for doc in documents])

            # Create the prompt for the LLM
            prompt = f"""Based on the following documents, please answer the question.
            If the answer cannot be found in the documents, say so.

            Documents:
            {context}

            Question: {question}

            Answer:"""

            # Get the answer from the LLM
            response = await llm.ainvoke(prompt)

            return CaseDocumentOutputSchema(answer=response, documents=documents)

        except Exception as e:
            logger.error(f"Error in RAG system: {str(e)}")
            return CaseDocumentOutputSchema(answer=f"Error: {str(e)}", documents=[])

    # Return the function info
    yield FunctionInfo.from_fn(
        fn=_answer_case_question,
        # TODO: add something like: Be sure to use the case_id from the context when making tool calls.
        description="Retrieves case-specific documents and uses an LLM to answer questions about them. ",
    )
