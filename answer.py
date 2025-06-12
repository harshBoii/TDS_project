from typing import List, Dict
from openai import OpenAI
from Embeddings import index, embed_text
import json
from pathlib import Path
import openai
import base64
import tempfile
from typing import List, Dict, Optional

def retrieve(question: str, top_k: int = 10) -> List[Dict]:
    q_emb = embed_text(question)
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    snippets = []
    for match in res["matches"]:
        # Load chunks from JSON file to get full text
        chunks = json.loads(Path("all_chunks.json").read_text(encoding="utf-8"))["chunks"]
        # Find matching chunk by ID
        chunk = next((c for c in chunks if c["id"] == match["id"]), None)
        if chunk:
            snippets.append({
                "text": chunk["text"],  # Use full text instead of just ID
                "metadata": chunk["metadata"],
                "source": chunk["source"]
            })
    return snippets


def build_prompt(snippets: List[Dict], question: str) -> str:
    # Debug: Print number of snippets received
    print(f"Building prompt with {len(snippets)} snippets")
    
    # Debug: Print first snippet metadata
    if snippets:
        print("First snippet metadata:", snippets[0]["metadata"])
    
    # Build context with source and text, adding debug info
    ctx_parts = []
    for i, s in enumerate(snippets):
        try:
            source = s["source"]
            text = s["text"]
            created_at = s["metadata"].get("created_at", "Unknown Date")
            thread = s["metadata"].get("topic", "Unknown Thread")
            author = s["metadata"].get("author")
            tutor = s["metadata"].get("tutor")
            url_path = s["metadata"].get("url")
            ctx_parts.append(f"Source: {source}\n{text}\nCreated at: {created_at}\nThread: {thread}\nAuthor: {author}\nTutor: {tutor}")
            # Debug: Print each snippet as it's processed
            print(f"Processed snippet {i+1}/{len(snippets)}")
        except KeyError as e:
            print(f"Error processing snippet {i}: Missing key {e}")
            continue
    
    ctx = "\n\n".join(ctx_parts)
    # print(ctx)
    
    # Debug: Print final context length
    print(f"Final context length: {len(ctx)} characters")
    
    prompt = f"""
You are an expert TDS tutor .Use the following excerpts to answer:

{ctx}

Question: {question}

Answer:
"""
    return prompt

chat_client = OpenAI(
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjEwMDIyODVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.QL8DL1JSynSX8fAicz_79Fy2aUEZnBQvyyk-Hf9jqSM",
    base_url="https://aipipe.org/openrouter/v1"
)


def answer_question(question: str, image_base64: Optional[str] = None) -> str:
    snippets = retrieve(question)
    prompt = build_prompt(snippets, question)
    
    # Build message payload
    messages = [
        {"role": "system", "content": "You will be given a question and several text excerpts (plus an optional image). Follow these rules when answering:\n\n1. Search the excerpts for the answer. If none of them contains the answer, reply “I don’t know.”\n2. If the first excerpt’s text exactly matches the question, and any other excerpt is authored by Carlton or Jivraj and both excerpts are from the same thread, then that latter excerpt is most likely key—prioritize it.\n3. Pay special attention to excerpts that discuss the same thread or topic as the question.\n4. If an image is provided, base your answer on both the image and the excerpts.\n5. Ensure your answer is fully relevant to the question."},
    ]

    if image_base64:
        # Save image temporarily and create a URL-like object
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp:
            tmp.write(base64.b64decode(image_base64))
            tmp.flush()
            # Many OpenAI-compatible APIs support base64 image input using `image` key directly:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{image_base64}"}}
                ]
            })
    else:
        messages.append({"role": "user", "content": prompt})

    # Make completion request
    resp = chat_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages
    )
    answer = resp.choices[0].message.content.strip()

    links = []
    for s in snippets[:10]:
        url_path = s["metadata"].get("url")
        links.append({
            "url": f"https://discourse.onlinedegree.iitm.ac.in{url_path}",
            "text": s["text"]
        })

    return {
        "answer": answer,
        "links": links
    }

if __name__ == "__main__":
    print(answer_question("The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?"))