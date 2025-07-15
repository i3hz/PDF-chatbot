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

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

if not os.getenv("SERPAPI_API_KEY"):
    raise ValueError("SERPAPI_API_KEY not found in .env file. Please add it.")


app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_FOLDER = 'vector_stores'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

conversations = {}

def get_agent_executor(chat_id: str):
    """
    Creates a conversational agent that can use a PDF retriever and web search as tools.
    """
    vector_store_path = os.path.join(VECTOR_STORE_FOLDER, chat_id)
    if not os.path.exists(vector_store_path):
        return None

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    pdf_retriever_tool = Tool(
        name="PDF_Document_Search",
        func=retrieval_qa_chain.invoke,
        description="Use this tool to answer questions based on the content of the uploaded PDF document. Do not use it for any other questions."
    )

    search = SerpAPIWrapper()
    search_tool = Tool(
        name="General_Web_Search",
        func=search.run,
        description="Use this tool for general knowledge questions, current events, or any topic that is not covered in the PDF document."
    )

    tools = [pdf_retriever_tool, search_tool]

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True helps in debugging

    return agent_executor


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
    Receives a question, gets the answer from the agent, and updates the conversation.
    """
    agent_executor = get_agent_executor(chat_id)
    chat_session = conversations.get(chat_id)

    if agent_executor and chat_session:
        chat_history = []
        for conv in chat_session['history']:
            chat_history.append(HumanMessage(content=conv['user']))
            chat_history.append(AIMessage(content=conv['bot']))

        result = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        answer = result.get('output', 'Sorry, I encountered an error.')
        
        chat_session['history'].append({'user': question, 'bot': answer})
    
    return RedirectResponse(url=f"/chat/{chat_id}", status_code=303)