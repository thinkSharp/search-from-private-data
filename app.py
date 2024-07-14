from question_answer_with_pdf_with_rag import DocumentHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Example usage:
doc_paths = ["data/vbresume.pdf"]
chroma_collection_name = "my_chroma_collection"
embedding_model_id = "amazon.titan-embed-text-v2:0"
llm_model_id = "anthropic.claude-v2:1"


document_handler = DocumentHandler(chroma_collection_name, embedding_model_id, llm_model_id)
rag_chain = document_handler.get_rag_chain(doc_paths=doc_paths)

response = rag_chain.invoke('Which universities did vishal attend?')

print(response)



