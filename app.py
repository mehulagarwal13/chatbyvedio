import os
import re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

state = {"chain": None, "video_id": None}


class LoadRequest(BaseModel):
    video_id: str


class AskRequest(BaseModel):
    question: str


def extract_video_id(video_input: str) -> str:
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)", video_input)
    return match.group(1) if match else video_input.strip()


@app.post("/load")
def load_video(req: LoadRequest):
    video_id = extract_video_id(req.video_id)

    try:
        ytt = YouTubeTranscriptApi()
        try:
            fetched = ytt.fetch(video_id, languages=["en", "en-US", "en-GB", "en-IN"])
        except (NoTranscriptFound, Exception):
            available = ytt.list(video_id)
            fetched = next(iter(available)).fetch()
        transcript = " ".join(
            chunk.text if hasattr(chunk, "text") else chunk["text"]
            for chunk in fetched
        )
    except (TranscriptsDisabled, VideoUnavailable) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch transcript: {type(e).__name__}: {e}")

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {type(e).__name__}: {e}")

    prompt = PromptTemplate(
        template="""You are a helpful assistant you only give responses based on the following video transcript context. If you don't know the answer, say you don't know. Always use all available context to answer.

{context}

Question: {question}""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    state["chain"] = chain
    state["video_id"] = video_id

    return {"status": "ok", "video_id": video_id, "chunks": len(chunks)}


@app.post("/ask")
def ask_question(req: AskRequest):
    if state["chain"] is None:
        raise HTTPException(status_code=400, detail="No video loaded. Call /load first.")
    answer = state["chain"].invoke(req.question)
    return {"answer": answer}


@app.get("/status")
def get_status():
    return {"loaded": state["chain"] is not None, "video_id": state["video_id"]}


# Serve frontend — must come last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
