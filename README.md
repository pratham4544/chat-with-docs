markdown
# Chat PDF

## Overview
This Streamlit application allows users to interactively ask questions from documents such as PDFs, text files, and CSVs. The application processes the uploaded files, extracts text from them, creates embeddings, and enables conversational interaction based on user questions.

## Installation
Ensure you have Python installed, then install the required packages:
```bash
pip install streamlit PyPDF2 pandas langchain langchain_google_genai google
```

## Usage
1. Upload your files by clicking on the "Upload your Files" button.
2. After uploading, click on the "Submit & Process" button to start processing the files.
3. Enter your question in the text input field provided.
4. The system will provide a reply based on the content of the uploaded files.

## Code Description
- The code utilizes Streamlit for building the web application interface.
- PyPDF2 library is used for extracting text from PDF files.
- Text processing and splitting are done using `RecursiveCharacterTextSplitter` from `langchain` package.
- `GoogleGenerativeAIEmbeddings` from `langchain_google_genai` is used for generating embeddings.
- The FAISS library is employed for creating a vector store.
- A conversational chain is constructed using `ChatGoogleGenerativeAI` for answering user questions.
- The application is configured to load Google API key from environment variables using `dotenv`.

## File Structure
- `README.md`: This file containing information about the application.
- `app.py`: The main Python script for the Streamlit application.

## Dependencies
- streamlit
- PyPDF2
- pandas
- langchain
- langchain_google_genai
- google

## Running the Application
Run the following command in the terminal:
```bash
streamlit run app.py
```

## Additional Notes
- Ensure you have set up the required environment variables, especially the `GOOGLE_API_KEY` for accessing Google Generative AI.
- This application is intended for interactive querying of documents and may require suitable computational resources for processing large documents efficiently.
```
