# book_embedder.py

import os
import chromadb
import fitz  # PyMuPDF
import uuid
from tkinter import Tk, filedialog
import requests

# Setup persistent ChromaDB client with Ollama embeddings
chroma_client = chromadb.PersistentClient(path="./chromadb_data")

# Use Ollama for embeddings (make sure Ollama is running!)
collection = chroma_client.get_or_create_collection(
    name="lixiviacion",
    metadata={"hnsw:space": "cosine"}
)

# Function to get embeddings from Ollama
def get_ollama_embedding(text, model="nomic-embed-text"):
    """Get embeddings from Ollama (must be running locally)"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text}
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Error from Ollama: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running! You can start it by running 'ollama serve' in a terminal.")
        return None

# Function to extract and chunk PDF text
def extract_chunks_from_pdf(file_path, chunk_size=500):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Split into overlapping chunks
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Main process
if __name__ == "__main__":
    print("=" * 60)
    print("    PDF TO EMBEDDINGS CONVERTER")
    print("=" * 60)
    print("\nThis tool will convert your PDF into a searchable database.")
    print("\nIMPORTANT: Make sure Ollama is running before continuing!")
    print("(If you haven't started Ollama, open a terminal and type: ollama serve)\n")

    input("Press ENTER to select your PDF file...")

    # Open file picker dialog
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    pdf_path = filedialog.askopenfilename(
        title="Select PDF file to convert",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )

    root.destroy()

    if not pdf_path:
        print("\nNo file selected. Exiting...")
        exit(0)

    if not os.path.exists(pdf_path):
        print(f"\nError: File not found: {pdf_path}")
        input("\nPress ENTER to exit...")
        exit(1)

    print(f"\nSelected file: {pdf_path}")
    print("\n[Step 1/3] Extracting text from PDF...")

    try:
        chunks = extract_chunks_from_pdf(pdf_path)
        print(f"✓ Extracted {len(chunks)} chunks from PDF.")
    except Exception as e:
        print(f"\n✗ Error reading PDF: {e}")
        input("\nPress ENTER to exit...")
        exit(1)

    print("\n[Step 2/3] Testing connection to Ollama...")
    test_embedding = get_ollama_embedding("test")
    if test_embedding is None:
        print("\n✗ Could not connect to Ollama!")
        print("\nPlease make sure:")
        print("1. Ollama is installed (download from: https://ollama.ai)")
        print("2. Ollama is running (open terminal and type: ollama serve)")
        print("3. You have the embedding model (type: ollama pull nomic-embed-text)")
        input("\nPress ENTER to exit...")
        exit(1)

    print("✓ Connected to Ollama successfully!")

    print(f"\n[Step 3/3] Creating embeddings and storing in database...")
    print("(This may take a few minutes depending on PDF size)\n")

    for i, chunk in enumerate(chunks, 1):
        embedding = get_ollama_embedding(chunk)
        if embedding:
            chunk_id = str(uuid.uuid4())
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[chunk_id]
            )
            # Show progress
            if i % 10 == 0 or i == len(chunks):
                print(f"  Progress: {i}/{len(chunks)} chunks processed...")
        else:
            print(f"\n✗ Error getting embedding for chunk {i}")
            print("Stopping process...")
            input("\nPress ENTER to exit...")
            exit(1)

    print("\n" + "=" * 60)
    print("✓ SUCCESS! Your PDF has been converted to embeddings!")
    print("=" * 60)
    print(f"\nDatabase location: {os.path.abspath('./chromadb_data')}")
    print(f"Total chunks stored: {len(chunks)}")
    print("\nYou can now use this database for semantic search!")
    input("\nPress ENTER to exit...")
