# from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import create_retrieval_chain
import fitz
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)

# function to extract text from pdfs
def funcExtractPDFText(filePDF):
    
    with open("tempResume.pdf","wb") as f:
        f.write(filePDF.getbuffer())
    strExtractedText=""
    filePDF=fitz.open("tempResume.pdf")
    for pg in filePDF:
        strExtractedText+=pg.get_text("text")
    return strExtractedText

# function to chunk pdf text to Document format for vector stores
def funcChunkTxtToDoc(instrResRawTxt):

    objSplitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n\n","\n","●","."," "])
    chunkedTxt=objSplitter.split_text(instrResRawTxt)
    docForVectorStore=[Document(page_content=chunk) for chunk in chunkedTxt]
    return docForVectorStore

# function to create FAISS retriever
def funcDefFAISSRetriever(indocForVectorStores):
    objFAISSVectorStore=FAISS.from_documents(documents=indocForVectorStores, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
    return objFAISSVectorStore.as_retriever(search_type="mmr")

# function to create ChromaDB retriever
def funcDefChromaRetriever(indocForVectorStores):
    objChromaVectorStore=Chroma.from_documents(documents=indocForVectorStores, embedding=OpenAIEmbeddings())
    return objChromaVectorStore.as_retriever(search_type="mmr")

# function to build RAG Chain using retriever, prompt, and llm
def funBuildRAGChain(inObjRetriever, inObjPrompt, inObjLLM):
    objRAGChain=create_retrieval_chain(
        retriever=inObjRetriever,
        combine_docs_chain=inObjPrompt | inObjLLM
    )
    return objRAGChain

# function to create Chroma vector store and Rag Chain
def funcDefRagChainChroma(objDoc,inObjPrompt,inObjLLM):
    objVectorStore=Chroma.from_documents(documents=objDoc, embedding=OpenAIEmbeddings())
    objRetriever=objVectorStore.as_retriever()
    objRagChain=create_retrieval_chain(
            retriever=objRetriever,
            combine_docs_chain=inObjPrompt | inObjLLM
            )
    return objRagChain

# function to invoke rag chain on question(s) asked
def funcGenResponse(inStrQuestion, inObjRagChain):

    logging.info(f"Question to GPT:{inStrQuestion}")
    dictResponse=inObjRagChain.invoke({"input":inStrQuestion})
    
    logging.info(f"Response by GPT:{dictResponse['answer'].content}")
    return dictResponse["answer"].content
    


    