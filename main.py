import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Text Chunking API")

# ─── Helper functions ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip HTML/Markdown and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str,chunk_size: int = 800,overlap: int = 160) -> List[str]:
    
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(' '.join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks

def html_to_text(html: str) -> str:
    """Strip HTML tags via BeautifulSoup."""
    return BeautifulSoup(html, 'html.parser').get_text(separator=' ')

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    file_name: str | None = None
    file_index: int | None = None
    post_id: int | None = None
    created_at: str | None = None
    topic: str | None = None
    url: str | None = None
    author: str | None = None
    tutor: bool | None = None

class Chunk(BaseModel):
    id: str
    source: str
    text: str
    metadata: ChunkMetadata

class ChunkResponse(BaseModel):
    chunks: List[Chunk]
    total_chunks: int

# ─── 1. Course‐folder loader ────────────────────────────────────────────────────

def load_course_folder_chunks(folder_path: Path,chunk_size: int = 800,overlap: int = 160) -> List[Chunk]:
    
    out: List[Chunk] = []
    md_files = sorted(folder_path.glob("*.md"))
    for file_idx, md_path in enumerate(md_files):
        raw = md_path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        for chunk_idx, chunk in enumerate(chunk_text(cleaned, chunk_size, overlap)):
            out.append(Chunk(
                id=f"course-file{file_idx}-{chunk_idx}",
                source="CourseContent",
                text=chunk,
                metadata=ChunkMetadata(
                    file_name=md_path.name,
                    file_index=file_idx,
                    url=f"https://tds.s-anand.net/#/{md_path.name.split('.')[0]}"
                )
            ))
    return out

# ─── 2. Discourse‐folder loader ─────────────────────────────────────────────────

def load_discourse_folder_chunks(folder_path: Path,chunk_size: int = 800,overlap: int = 160) -> List[Chunk]:
    
    out: List[Chunk] = []
    json_files = sorted(folder_path.glob("*.json"))
    for file_idx, json_file in enumerate(json_files):
        raw = json_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        posts = data.get("post_stream", {}).get("posts", [])
        for post in posts:
            post_id = post["id"]
            created = post.get("created_at")
            topic = post.get("topic_slug") or post.get("name")
            txt = html_to_text(post.get("cooked", ""))
            cleaned = clean_text(txt)
            author = post.get("name")
            url = post.get("post_url")
            tutor = False  
            if author == "Carlton D'Silva":
                tutor=True
            for chunk_idx, chunk in enumerate(chunk_text(cleaned, chunk_size, overlap)):
                out.append(Chunk(
                    id=f"disc-{file_idx}-{post_id}-{chunk_idx}",
                    source="Discourse",
                    text=chunk,
                    metadata=ChunkMetadata(
                        file_name=json_file.name,
                        post_id=post_id,
                        created_at=created,
                        topic=topic,
                        url=url,
                        author=author,
                        tutor=tutor
                    )
                ))
    return out

# ─── FastAPI Routes ───────────────────────────────────────────────────────────

@app.get("/chunks/course", response_model=ChunkResponse)
async def get_course_chunks(
    folder_path: str = "Course_MarkDown",
    chunk_size: int = 800,
    overlap: int = 160
):
    try:
        chunks = load_course_folder_chunks(Path(folder_path), chunk_size, overlap)
        return ChunkResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks/discourse", response_model=ChunkResponse)
async def get_discourse_chunks(
    folder_path: str = "discourse_json",
    chunk_size: int = 800,
    overlap: int = 160
):
    try:
        chunks = load_discourse_folder_chunks(Path(folder_path), chunk_size, overlap)
        return ChunkResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks/all", response_model=ChunkResponse)
async def get_all_chunks(
    course_folder: str = "Course_MarkDown",
    discourse_folder: str = "discourse_json",
    chunk_size: int = 800,
    overlap: int = 160
):
    try:
        course_chunks = load_course_folder_chunks(Path(course_folder), chunk_size, overlap)
        discourse_chunks = load_discourse_folder_chunks(Path(discourse_folder), chunk_size, overlap)
        all_chunks = course_chunks + discourse_chunks
        # Write JSON with null fields included
        out = {"chunks": [c.dict(exclude_none=False) for c in all_chunks]}
        Path("all_chunks.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2)
        )
        return ChunkResponse(chunks=all_chunks, total_chunks=len(all_chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
from answer import answer_question

@app.post("/api")
async def handle_question(question_data: dict):
    question = question_data.get("question")
    image = question_data.get("image")  # Optional for future image support

    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    try:
        result = answer_question(question,image_base64=image)
        print(f"Result: {result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
