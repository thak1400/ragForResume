from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
import os
import fitz

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "key"
os.environ["LANGSMITH_PROJECT"]="ragAppForDocs"
os.environ["OPENAI_API_KEY"] = "key"

# objLCPrompt=hub.pull("rlm/rag-prompt")
# objCustPrompt=ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
objResAstPrompt=Client().pull_prompt("rag-for-resume1")

objLLM=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def funcExtractPDFText(strPDFPath):
    
    with open("tempResume.pdf","wb") as f:
        f.write(strPDFPath.getbuffer())

    strExtractedText=""
    filePDF=fitz.open("tempResume.pdf")
    for pg in filePDF:
        strExtractedText+=pg.get_text("text")
    return strExtractedText

def funcDefRagChain(objDoc):

    objVectorStore=Chroma.from_documents(documents=objDoc, embedding=OpenAIEmbeddings())
    objRetriever=objVectorStore.as_retriever()
    objRagChain=create_retrieval_chain(
            retriever=objRetriever,
            combine_docs_chain=objResAstPrompt | objLLM
            )
    return objRagChain

def funcGenResponse(inStrQuestion, inObjRagChain):
    
    dictResponse=inObjRagChain.invoke({"input":inStrQuestion})
    return dictResponse["answer"].content
    


    