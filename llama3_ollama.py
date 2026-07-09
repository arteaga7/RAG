from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
import requests
import json
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# ChromaDB setup with OpenAI embeddings
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
collection = chroma_client.get_or_create_collection("demo_collection", embedding_function=embedding_function)

# Function to call local Ollama model like OpenAI
def ollama_chat_completion(model: str, system_prompt: str, user_prompt: str) -> str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    chunks = []
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "message" in data:
                    chunks.append(data["message"]["content"])
            except json.JSONDecodeError:
                continue
    return "".join(chunks)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    context = None

    if request.method == "POST":
        question = request.form["question"]

        # Search ChromaDB for relevant content
        results = collection.query(query_texts=[question], n_results=3)
        retrieved_docs = results["documents"][0]
        context = "\n".join(retrieved_docs)

        # Ask LLaMA 3 via Ollama
        answer = ollama_chat_completion(
            model="llama3",
            system_prompt="You are a helpful assistant. Use the provided context to answer the user's question clearly and concisely. You may paraphrase or synthesize the context, but do not invent information.",
            user_prompt=(
                f"Answer the question based on the following context. If multiple passages are relevant, combine them meaningfully.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        )

    return render_template("index.html", answer=answer, context=context)

if __name__ == "__main__":
    app.run(debug=True)