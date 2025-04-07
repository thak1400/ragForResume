import streamlit as st
from ragChain import funcDefRagChain, funcExtractPDFText,funcGenResponse
from langchain.schema import Document

def main():
    
    st.title("Resume Assistant")
    
    fileUploaded=st.file_uploader("Upload your resume (PDF)", type="pdf")
    
    if fileUploaded is not None:
        strResRawTxt=funcExtractPDFText(fileUploaded)
        docFormatted=[Document(page_content=strResRawTxt)]

        objRagChain=funcDefRagChain(docFormatted)

        strQuestion=st.text_input("Ask about the resume:")

        if strQuestion:
            strAnswer=funcGenResponse(strQuestion,objRagChain)
            st.write("Answer:",strAnswer)

if __name__=="__main__":
    main()
            