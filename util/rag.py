import numpy as np
import litellm

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embeddings(texts, model="nvidia_nim/nvidia/nv-embed-v1"):
    embeddings = []
    for text in texts:
        try:
            response = litellm.embedding(model=model, input=text, encoding_format="float")
            embeddings.append(response.data[0]['embedding'])
        except Exception as e:
            print(f"Error getting embedding: {e}")
            embeddings.append(None)
    return embeddings

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def search_relevant_context(query, text, top_k=3, model="nvidia_nim/nvidia/nv-embed-v1"):
    chunks = chunk_text(text)
    if not chunks:
        return ""

    chunk_embeddings = get_embeddings(chunks, model=model)
    query_embedding = get_embeddings([query], model=model)[0]

    if query_embedding is None:
        return chunks[0]

    scored_chunks = []
    for i, emb in enumerate(chunk_embeddings):
        if emb is not None:
            score = cosine_similarity(query_embedding, emb)
            scored_chunks.append((score, chunks[i]))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
    return "\n...\n".join(top_chunks)
