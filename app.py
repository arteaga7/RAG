from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import ollama

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Persistent ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
# embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    context = None

    # Get all available collections
    collections = chroma_client.list_collections()
    collection_names = [c.name for c in collections]
    selected_collection = request.form.get("collection") or (
        collection_names[0] if collection_names else "")

    if selected_collection:
        collection = chroma_client.get_or_create_collection(
            selected_collection)

        if request.method == "POST":
            question = request.form["question"]

            # Query embeddings
            # results = collection.query(query_texts=[question], n_results=10)
            query_embedding = ollama.embeddings(
                model="nomic-embed-text", prompt=question
            )["embedding"]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            retrieved_docs = results["documents"][0]
            context = "\n".join(retrieved_docs)

            # Call GPT
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Use the provided context to answer the user's question clearly and concisely. "
                            "You may paraphrase or synthesize the context, but do not invent information."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Answer the question based on the following context. If multiple passages are relevant, combine them meaningfully.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        )
                    }
                ]
            )
            answer = response.choices[0].message.content

    return render_template("index.html",
                           answer=answer,
                           context=context,
                           collections=collection_names,
                           selected_collection=selected_collection)


if __name__ == "__main__":
    app.run(debug=True)
