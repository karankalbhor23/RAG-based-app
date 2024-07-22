import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_chain = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.post("/process/")
async def process_pdfs(files: list[UploadFile] = File(...)):
    pdf_docs = [file.file for file in files]
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    global conversation_chain
    conversation_chain = get_conversation_chain(vectorstore)
    return {"message": "PDFs processed successfully"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global conversation_chain
    if conversation_chain is None:
        return {"error": "No conversation chain available. Please process PDFs first."}
    response = conversation_chain({'question': question})
    chat_history = response['chat_history']
    return {"chat_history": chat_history}

@app.get("/")
async def main():
    content = """
    <html>
        <head>
            <title>Chat with multiple PDFs</title>
        </head>
        <body>
            <h1>Upload PDFs and Ask Questions</h1>
            <form action="/process/" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <input type="submit" value="Process">
            </form>
            <br>
            <form action="/ask/" method="post">
                <input name="question" type="text" placeholder="Ask a question about your documents">
                <input type="submit" value="Ask">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == '__main__':
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
