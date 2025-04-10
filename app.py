import streamlit as st
from ragChain import funcDefRagChainChroma, funcExtractPDFText,funcGenResponse,funcChunkTxtToDoc, funcDefFAISSRetriever, funcDefChromaRetriever, funBuildRAGChain
from langchain.schema import Document
import logging
import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain.retrievers import BM25Retriever, EnsembleRetriever

logging.basicConfig(level=logging.INFO)

#environment variable
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "add-your-api-key"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]="ragAppForDocs"
os.environ["OPENAI_API_KEY"] = "add-your-api-key"


def main():
    
    st.title("Resume Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]

    fileUploaded=st.file_uploader("Upload your resume (PDF)", type="pdf")
    
    stVectorChoice=st.radio("Choose Retriever",options=["ChromaDB","FAISS","BM25Retriever","Ensemble"], index=0)
    stLLM=st.radio("Choose LLM",options=["Claude3","ChatGPT"], index=0)

    if fileUploaded is not None:

        #extracting text from pdf
        strResRawTxt=funcExtractPDFText(fileUploaded)
        print(strResRawTxt)
        
        # converting raw text to chunks
        docForVectorStores=funcChunkTxtToDoc(strResRawTxt)

        # selecting retriever based on user's choice
        if stVectorChoice=="FAISS":
            objRetriever=funcDefFAISSRetriever(docForVectorStores)
        elif stVectorChoice=="ChromaDB":
            objRetriever=funcDefChromaRetriever(docForVectorStores)
        elif stVectorChoice=="BM25Retriever":
            objRetriever=BM25Retriever.from_documents(docForVectorStores)
        else:
            objRetriever=EnsembleRetriever(
                retrievers=[funcDefFAISSRetriever(docForVectorStores),funcDefChromaRetriever(docForVectorStores), BM25Retriever.from_documents(docForVectorStores)],
                weights=[0.34,0.33,0.33]
                )
            
        #custom LC prompt for resumes
        objResAstPrompt=Client().pull_prompt("rag-for-resume1")

        # selecting LLM based on user's choice
        if stLLM=="ChatGPT":
            objLLM=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        else:
            objLLM=ChatAnthropic(model="claude-3-opus-20240229", api_key="add-your-api-key")

        objRagChain=funBuildRAGChain(objRetriever, objResAstPrompt, objLLM)
        
        # #formatting for Chroma
        # docFormatted=[Document(page_content=strResRawTxt)]

        # #creating chain
        # objRagChain=funcDefRagChainChroma(docFormatted,objResAstPrompt,objLLM)
        
        # displaying chat history
        for message in st.session_state.chat_history:
            if isinstance(message,HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message,AIMessage):
                with st.chat_message("assistnant"):
                    st.write(message.content)

        strQuestion=st.text_input("Ask about the resume:")

        if strQuestion:
            
            # appending user's asked question to chat history
            st.session_state.chat_history.append(HumanMessage(content=strQuestion))
            with st.chat_message("user"):
                st.write(strQuestion)

            # querying RAG Chain
            strAnswer=funcGenResponse(strQuestion,objRagChain)

            # appending AI's answers to chat history
            st.session_state.chat_history.append(AIMessage(content=strAnswer))        
            with st.chat_message("assistant"):
                st.write(strAnswer)

if __name__=="__main__":
    main()
            