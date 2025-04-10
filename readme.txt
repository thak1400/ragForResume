# Streamlit RAG App

This is a Streamlit application that utilizes LangChain and OpenAI to build a Retrieval-Augmented Generation (RAG) model. The app allows users to upload resumes (PDF format) and asks questions related to the resume. It uses a combination of Document Embeddings and LLM (Large Language Model) to generate responses based on the context of the uploaded document.

## Features

- Resume Upload: Users can upload their resumes in PDF format.
- Question Answering: Once the resume is uploaded, users can ask questions related to the content of the resume.
- Retrieval-Augmented Generation (RAG): The app uses RAG to retrieve relevant information from the resume and generate responses based on it.
- Integration with OpenAI: The app utilizes the OpenAI API for generating responses using models like `gpt-3.5-turbo`.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/thak1400/ragForResume.git

2. cd ragForResume

3. Add these variables to rgaChain.py
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "YOUR KEY
os.environ["LANGSMITH_API_KEY"] = "key"
os.environ["LANGSMITH_PROJECT"]="ragAppForDocs"
os.environ["OPENAI_API_KEY"] = "YOUR KEY

4. streamlit run app.py
