# Chat with your PDF - A RAG Application
![Application Screenshot](https://raw.githubusercontent.com/i3hz/PDF-chatbot/main/assets/screenshot.png)
  
This project is a web application that allows you to upload a PDF document and have an interactive chat with it. It uses a powerful AI technique called **Retrieval-Augmented Generation (RAG)** to provide answers based exclusively on the content of your document.

The application is built with **FastAPI** for the backend, **LangChain** for orchestrating the AI logic, and **OpenAI's** models for embeddings and language generation.

## Features

-   **Simple Web Interface**: Easy-to-use UI for uploading PDFs and chatting.
-   **Secure & Private**: Your document is processed in a unique, isolated session. The AI model only answers based on the provided document, not its general knowledge.
-   **Context-Aware Conversations**: The AI remembers the context of the chat within a session.
-   **Source-Grounded Answers**: The answers are generated directly from the text within your PDF.
-   **Powered by State-of-the-Art AI**: Utilizes OpenAI's powerful embedding and `gpt-4-turbo` models for high-quality results.

## How It Works: The RAG Architecture

This application is a practical implementation of the Retrieval-Augmented Generation (RAG) pattern. Here’s a high-level overview of the process:

1.  **Ingestion (Uploading a PDF)**:
    *   You upload a PDF file.
    *   The system extracts all the text from the PDF.
    *   The text is broken down into smaller, manageable chunks.
    *   Each chunk is converted into a numerical representation (an "embedding") using an OpenAI model. These embeddings capture the semantic meaning of the text.
    *   These embeddings are stored in a highly efficient **FAISS vector store**, which acts as a searchable knowledge base for your document.

2.  **Retrieval & Generation (Asking a Question)**:
    *   You ask a question in the chat interface.
    *   The system converts your question into an embedding.
    *   It then searches the FAISS vector store to find the text chunks from your original PDF that are most semantically similar to your question.
    *   These relevant chunks (the "context") are combined with your original question into a detailed prompt.
    *   This prompt is sent to a powerful language model (GPT-4 Turbo), which generates a coherent, human-like answer based *only* on the information provided in the context.

This "retrieve-then-read" approach ensures that the answers are accurate, relevant, and grounded in the source document.


## Tech Stack

-   **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
-   **AI Orchestration**: [LangChain](https://www.langchain.com/)
-   **LLM & Embeddings**: [OpenAI](https://openai.com/) (`gpt-4-turbo`, `text-embedding-ada-002`)
-   **Vector Store**: [FAISS (Facebook AI Similarity Search)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
-   **Frontend**: [Jinja2 Templates](https://jinja.palletsprojects.com/) with basic HTML/CSS.
-   **Environment Management**: [python-dotenv](https://github.com/theskumar/python-dotenv)

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.9+
-   An OpenAI API key.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*(Note: If you don't have a `requirements.txt` file yet, you can create one from your environment using `pip freeze > requirements.txt`)*

### 4. Configure Environment Variables

You need to provide your OpenAI API key. Create a file named `.env` in the root of the project directory and add your key:

```.env
OPENAI_API_KEY="sk-YourSecretOpenAI_KeyGoesHere"
```

The application uses `python-dotenv` to load this key automatically.

### 5. Run the Application

Now you are ready to start the FastAPI server using `uvicorn`.

```bash
uvicorn main:app --reload
```

-   `main`: The file `main.py` (the Python module).
-   `app`: The `FastAPI()` object created in `main.py`.
-   `--reload`: This flag makes the server restart automatically after code changes, which is great for development.

The server will start, and you can access the application in your web browser at:

**http://127.0.0.1:8000**

## How to Use the App

1.  **Navigate to the Homepage**: Open `http://127.0.0.1:8000` in your browser.
2.  **Upload a PDF**: Click the "Choose File" button, select a PDF from your computer, and click "Upload".
3.  **Wait for Processing**: The app will process the PDF (this may take a few moments depending on the file size) and then redirect you to a new chat page.
4.  **Start Chatting**: You are now in a unique chat session for your document. Type your questions into the input box at the bottom and press Enter or click "Ask".
5.  **View Responses**: The AI's response will appear in the chat window. The conversation history is maintained for your session.

## Future Improvements

-   **Persistent Chat History**: Replace the in-memory `conversations` dictionary with a database like SQLite or Redis to persist chat history across server restarts.
-   **Asynchronous Processing**: Move the PDF ingestion pipeline to a background task (e.g., using `FastAPI`'s `BackgroundTasks` or Celery) to prevent the UI from hanging on large file uploads.
-   **Streaming Responses**: Implement response streaming from the LLM to display the answer word-by-word for a more interactive user experience.
-   **Source Highlighting**: Enhance the UI to show which chunks of the original document were used to generate the answer.
-   **Dockerize the Application**: Create a `Dockerfile` to make deployment easier and more consistent.
