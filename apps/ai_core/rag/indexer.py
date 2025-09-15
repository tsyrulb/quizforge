import os, uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PERSIST = os.getenv("RAG_PERSIST", "./rag_store")

class RAGIndex:
    def __init__(self, collection_name="docs"):
        self.client = chromadb.Client(Settings(persist_directory=PERSIST))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space":"cosine"})
        self.embedder = SentenceTransformer(MODEL_NAME)

    def add_documents(self, docs):
        # docs: [{id?, title, text, source}]
        ids, texts, metadatas = [], [], []
        for d in docs:
            did = d.get("id") or str(uuid.uuid4())
            ids.append(did)
            texts.append(d["text"])
            metadatas.append({"title": d.get("title",""), "source": d.get("source","")})
        embs = self.embedder.encode(texts, convert_to_numpy=True).tolist()
        self.collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metadatas)
        self.client.persist()

    def retrieve(self, query: str, top_k=6):
        qemb = self.embedder.encode([query], convert_to_numpy=True).tolist()[0]
        res = self.collection.query(query_embeddings=[qemb], n_results=top_k)
        chunks = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            chunks.append(f'{meta.get("title","")}: {doc[:1200]}')
        return "\n---\n".join(chunks) if chunks else ""
