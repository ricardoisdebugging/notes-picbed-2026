import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import uvicorn

# ====== 配置 ======
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
API_KEY = "local-key"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 加载模型 ======
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
rerank_model.to(DEVICE)
rerank_model.eval()

# ====== FastAPI ======
app = FastAPI()

class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

# ====== API KEY 验证 ======
def verify_key(authorization: str):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ====== Embedding API ======
@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, authorization: str = Header(...)):
    verify_key(authorization)

    embeddings = embedding_model.encode(
        req.input,
        normalize_embeddings=True
    )

    return {
        "data": [
            {"embedding": emb.tolist(), "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "model": EMBEDDING_MODEL_NAME
    }

# ====== Rerank API ======
@app.post("/v1/rerank")
def rerank(req: RerankRequest, authorization: str = Header(...)):
    verify_key(authorization)

    pairs = [[req.query, doc] for doc in req.documents]

    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        scores = rerank_model(**inputs).logits.squeeze(-1)
        scores = scores.cpu().numpy()

    return {
        "scores": scores.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)