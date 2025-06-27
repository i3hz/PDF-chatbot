import os
import uuid
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_FOLDER = 'vector_stores'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

conversations = {}

def get_qa_chain(chat_id: str):
    """
    Looks up the vector store for the given chat_id and creates a QA chain.
    """
    vector_store_path = os.path.join(VECTOR_STORE_FOLDER, chat_id)
    if not os.path.exists(vector_store_path):
        return None

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

@app.get("/", response_class=HTMLResponse, name="index")
async def read_root(request: Request):
    """
    Serves the initial page with the file upload form.
    """
    return templates.TemplateResponse("index.html", {"request": request, "chat_id": None})

@app.post("/upload", response_class=RedirectResponse)
async def handle_upload(pdf_file: UploadFile = File(...)):
    """
    Handles PDF upload, processing, and redirection to the chat page.
    """
    if not pdf_file.filename.endswith('.pdf'):
        return RedirectResponse(url="/", status_code=303)

    chat_id = str(uuid.uuid4())
    
    filepath = os.path.join(UPLOAD_FOLDER, f"{chat_id}.pdf")
    with open(filepath, "wb") as f:
        content = await pdf_file.read()
        f.write(content)

    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store_path = os.path.join(VECTOR_STORE_FOLDER, chat_id)
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(vector_store_path)

    conversations[chat_id] = {'pdf_filename': pdf_file.filename, 'history': []}

    return RedirectResponse(url=f"/chat/{chat_id}", status_code=303)

@app.get("/chat/{chat_id}", response_class=HTMLResponse)
async def chat_page(request: Request, chat_id: str):
    """
    Displays the chat interface for a specific document.
    """
    chat_session = conversations.get(chat_id)
    if not chat_session:
        return RedirectResponse(url="/")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_id": chat_id,
        "pdf_filename": chat_session['pdf_filename'],
        "conversation": chat_session['history']
    })


@app.post("/ask/{chat_id}", response_class=RedirectResponse)
async def ask_question(chat_id: str, question: str = Form(...)):
    """
    Receives a question, gets the answer from LangChain, and updates the conversation.
    """
    qa_chain = get_qa_chain(chat_id)
    chat_session = conversations.get(chat_id)

    if qa_chain and chat_session:
        result = qa_chain.invoke({"query": question})
        
        answer = result['result']
        
        # Update conversation history
        chat_session['history'].append({'user': question, 'bot': answer})
    
    return RedirectResponse(url=f"/chat/{chat_id}", status_code=303)