# Chat by Video

A FastAPI-based web app that lets you chat with any YouTube video using AI. Paste a YouTube URL, load the transcript, and ask questions — powered by Groq's LLaMA 3 and HuggingFace embeddings.

## Features

- Fetches YouTube transcripts automatically (supports multiple languages)
- Chunks and indexes the transcript using FAISS vector store
- Answers questions using RAG (Retrieval-Augmented Generation)
- Clean single-page frontend served from FastAPI
- Deployable to Railway in minutes

## Tech Stack

| Layer | Library |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Groq (LLaMA 3.3 70B) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Transcripts | youtube-transcript-api |
| Orchestration | LangChain |

## Getting Started (Local)

**1. Clone the repo**
```bash
git clone https://github.com/mehulagarwal13/chatbyvedio.git
cd chatbyvedio
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

**5. Run the app**
```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/load` | Load a YouTube video by URL or ID |
| `POST` | `/ask` | Ask a question about the loaded video |
| `GET` | `/status` | Check if a video is currently loaded |

### Example

```bash
# Load a video
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"video_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this video about?"}'
```

## Deploy to Railway

1. Fork/push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select this repo — Railway auto-detects Python and uses the `Procfile`
4. In your Railway service → **Variables**, add:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
5. Deploy — your app will be live at a public Railway URL

## Project Structure

```
chatbyvedio/
├── app.py            # FastAPI backend
├── static/
│   └── index.html    # Frontend UI
├── requirements.txt  # Python dependencies
├── Procfile          # Railway start command
└── .env              # Local secrets (not committed)
```

## License

MIT
