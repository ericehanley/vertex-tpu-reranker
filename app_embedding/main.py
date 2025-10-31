import os
from contextlib import asynccontextmanager
from typing import List

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import BatchEncoding

# A dictionary to hold the model components.
model_handler = {}

# --- CHOOSE YOUR MODEL ---
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'
# MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'


# --- HELPER FUNCTION FOR POOLING ---
def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup: load, separate, optimize, compile, and warm up the model.
    """
    print(f"🚀 Server starting up with model: {MODEL_NAME}")
    try:
        device = xm.xla_device()
        print(f"Successfully acquired TPU device: {device}")

        trust_remote = 'mxbai' in MODEL_NAME
        model_container = SentenceTransformer(
            MODEL_NAME,
            device='cpu',
            trust_remote_code=trust_remote
        )
        
        tokenizer = model_container.tokenizer
        model = model_container[0]  # Access the pure transformer model
        model.to(device)
        model.eval()
        print(f"Model loaded in float32 precision and set to eval mode.")

        print("Compiling the core model with the 'openxla' backend...")
        compiled_model = torch.compile(model, backend="openxla")
        
        model_handler["compiled_model"] = compiled_model
        model_handler["tokenizer"] = tokenizer
        print("Model compilation complete.")

        print("Starting end-to-end model warm-up...")
        warmup_batch_sizes = [1, 2, 4, 8, 16, 32] 
        dummy_input = ["This is a warmup sentence."]

        for batch_size in warmup_batch_sizes:
            print(f"   - Warming up for batch size: {batch_size}...")
            
            tokenized_input = tokenizer(
                dummy_input * batch_size,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            
            with torch.no_grad():

                model_output = compiled_model(tokenized_input)
                
                embeddings = mean_pooling(model_output, tokenized_input['attention_mask'])

            _ = embeddings[0].cpu()

        print("✅ Warm-up complete. Endpoint is fully ready for traffic.")

    except Exception as e:
        print(f"❌ An error occurred during startup or warm-up: {e}")
        model_handler.clear()

    yield

    print("🔻 Server shutting down...")
    model_handler.clear()


app = FastAPI(title="Sentence Embedding API", lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    instances: List[str] = Field(..., example=["This is a test sentence."])

class EmbeddingResponse(BaseModel):
    predictions: List[List[float]]


@app.get(os.environ.get('AIP_HEALTH_ROUTE', '/health'), status_code=200)
def health():
    return {"status": "healthy" if "compiled_model" in model_handler and "tokenizer" in model_handler else "unhealthy"}


@app.post(
    os.environ.get('AIP_PREDICT_ROUTE', '/predict'),
    response_model=EmbeddingResponse
)
async def predict(request: EmbeddingRequest):
    compiled_model = model_handler.get("compiled_model")
    tokenizer = model_handler.get("tokenizer")
    
    if not compiled_model or not tokenizer:
        return {"error": "Model not loaded or warm-up failed"}, 503
    
    device = next(compiled_model.parameters()).device

    tokenized_input = tokenizer(
        request.instances,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():

        model_output = compiled_model(tokenized_input)
        
        embeddings = mean_pooling(model_output, tokenized_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return {"predictions": embeddings.cpu().tolist()}