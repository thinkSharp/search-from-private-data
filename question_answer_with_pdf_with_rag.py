from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import chromadb


class DocumentHandler:
    def __init__(self, chroma_collection_name, embedding_model_id,
                  llm_model_id, chunk_size=1000, chunk_overlap=200):
        self.chroma_collection_name = chroma_collection_name
        self.embedding_model_id = embedding_model_id
        self.llm_model_id = llm_model_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=chroma_collection_name)
        self.embeddings_model = BedrockEmbeddings(credentials_profile_name='default', model_id=embedding_model_id)
        self.llm = self._initialize_llm(llm_model_id)

    def get_text_from_pdf(self, doc_paths):
        for path in doc_paths:
            reader = PdfReader(path)
            text = ' '.join(doc.extract_text() for doc in reader.pages)
        return text
    

    def load_split_and_store_documents_with_embeddings(self, doc_paths):
        split_documents = self._split_documents(doc_paths=doc_paths)
        vectorstore = Chroma.from_documents(documents=split_documents, embedding=self.embeddings_model)
        retriver = vectorstore.as_retriever()
        return retriver

    
    def _split_documents(self, doc_paths):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        split_documents = []
        for path in doc_paths:
            loader = PyPDFLoader(path)
            doc = loader.load()
            splits = splitter.split_documents(doc)
            split_documents.extend(splits)
        return split_documents



    def _initialize_llm(self, llm_model_id):
        model_kwargs = {
            "max_tokens_to_sample": 2048,
            "stop_sequences": ['\\n\\nHuman:'],
            "temperature": 0,
            "top_p":0.9,
            "top_k":200
        }
        return BedrockLLM(credentials_profile_name='default',
                           model_id=llm_model_id, model_kwargs=model_kwargs)

    def get_rag_chain(self, doc_paths):
        retriever = self.load_split_and_store_documents_with_embeddings(doc_paths)
        message = """
                    Answer this question using the provided context only. If don't know say I don't have it.

                    Context: {context}

                    {question}
                    """

        prompt = ChatPromptTemplate.from_messages(
            [ ("human", message) ]
            )

        rag_chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt | self.llm
        return rag_chain
    

