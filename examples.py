"""
📦 Examples of Embedders and Their Vector Sizes

| Embedder Type             | Model Name                         | Output Vector Size |
|---------------------------|------------------------------------|--------------------|
| Word Embedder             | Word2Vec, GloVe                    | Fixed (e.g., 300)  |
| Sentence Embedder         | Sentence-BERT (SBERT)              | Fixed (e.g., 768)  |
| Transformer Token Embedder| BERT Base                          | 768 per token      |
| Document Embedder         | OpenAI `text-embedding-3-small`    | 1536 (default)     |

1. Contextual
2. Token or Sentence Embeddings
2a. Token-level Embeddings
2b. Sentence-level Embeddings

Notes:
------
- Each embedder type produces **fixed-size** vectors determined by the model architecture.
- Word embedders embed one word at a time.
- Sentence embedders encode the whole sentence as one vector.
- Token embedders (like BERT) give a vector per token (word/subword).
- Some models (e.g., BERT, GPT) require pooling to get sentence/document embeddings.
"""