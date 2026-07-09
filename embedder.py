# embedder.py

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from dotenv import load_dotenv

# Load .env variables (for OPENAI_API_KEY)
load_dotenv()

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path="./chromadb_data")

# Create or get collection with embedding support
embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
collection = client.get_or_create_collection("demo_collection", embedding_function=embedding_function)

# Sample texts to embed
documents = [
    """The Eiffel Tower, completed in 1889 for the World's Fair, stands as an iconic symbol of Paris and a feat of architectural engineering. Gustave Eiffel, the man behind its design, initially faced criticism, but today the tower attracts over 7 million visitors each year. It stands at 324 meters tall and was the world's tallest structure until 1930.""",

    """Python is a versatile, high-level programming language created by Guido van Rossum in the early 1990s. It's known for its readable syntax, wide range of libraries, and suitability for web development, automation, data science, and artificial intelligence. Its philosophy emphasizes code readability and developer productivity.""",

    """The Amazon rainforest spans across nine South American countries and is often called the 'lungs of the Earth' due to its massive oxygen output. It is home to over 3 million species of plants and animals and plays a critical role in regulating the global climate. Deforestation and climate change pose major threats to this ecosystem.""",

    """Founded by Elon Musk, SpaceX has revolutionized the aerospace industry with its focus on reusable rockets, cost efficiency, and ambitious projects like Mars colonization. In 2020, SpaceX made history by launching NASA astronauts into orbit aboard a privately built spacecraft, the Crew Dragon.""",

    """Photosynthesis is a biochemical process in plants, algae, and some bacteria that converts sunlight, carbon dioxide, and water into glucose and oxygen. This process takes place in the chloroplasts and is vital not just for plant growth, but also for maintaining the Earth's oxygen balance and supporting the food chain.""",

    """Arturo Medina is a versatile developer and technical innovator working at LedgerFi. He actively builds intelligent systems using Python, Flask, and AI APIs, often integrating technologies like OpenAI, ChromaDB, and cloud platforms such as Heroku and AWS. Arturo has hands-on experience in transaction classification, chatbot design, and vector embedding-based search. He is driven by real-world applications of machine learning, especially in finance and bookkeeping. With a pragmatic and problem-solving mindset, Arturo often bridges the gap between backend engineering and AI-powered automation. His work combines technical depth with a strong focus on building user-facing solutions that are scalable, efficient, and intelligent.""",
]

# Add texts to collection (use unique IDs)
for idx, text in enumerate(documents):
    collection.add(documents=[text], ids=[f"doc_{idx}"])

print("Documents embedded and stored permanently in ChromaDB.")