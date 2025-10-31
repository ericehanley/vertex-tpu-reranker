import os
import traceback
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
#MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'


# --- HELPER FUNCTION FOR POOLING ---
def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output['token_embeddings']
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
        
        # --- CHANGE #1: Get the model's max sequence length ---
        max_seq_len = model_container.max_seq_length
        print(f"Model max sequence length set to: {max_seq_len}")

        tokenizer = model_container.tokenizer
        model = model_container[0]
        model.to(device)
        model.eval()
        print(f"Model loaded in float32 precision and set to eval mode.")

        print("Compiling the core model with the 'openxla' backend...")
        compiled_model = torch.compile(model, backend="openxla")

        model_handler["compiled_model"] = compiled_model
        model_handler["tokenizer"] = tokenizer
        model_handler["max_seq_len"] = max_seq_len # Store for use in predict
        print("Model compilation complete.")

        print("Starting end-to-end model warm-up...")
        warmup_batch_sizes = [1, 2, 4, 8]
        dummy_input = ["This is a warmup sentence."]

        for batch_size in warmup_batch_sizes:
            print(f"     - Warming up for batch size: {batch_size}...")

            # --- CHANGE #2: Modify tokenizer call for static shapes ---
            tokenized_input = tokenizer(
                dummy_input * batch_size,
                padding='max_length',  # Use max_length padding
                truncation=True,
                max_length=max_seq_len,  # Specify the length
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                model_output = compiled_model(tokenized_input)
                embeddings = mean_pooling(model_output, tokenized_input['attention_mask'])
            _ = embeddings[0].cpu()

        print("✅ Warm-up complete. Endpoint is fully ready for traffic.")

    except Exception as e:
        print(f"❌ An error occurred during startup or warm-up:")
        traceback.print_exc()
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
    max_seq_len = model_handler.get("max_seq_len")

    if not all([compiled_model, tokenizer, max_seq_len]):
        return {"error": "Model not loaded or warm-up failed"}, 503

    device = xm.xla_device()

    # --- CHANGE #3: Modify tokenizer call for static shapes here too ---
    tokenized_input = tokenizer(
        request.instances,
        padding='max_length', # Use max_length padding
        truncation=True,
        max_length=max_seq_len, # Specify the length
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        model_output = compiled_model(tokenized_input)
        embeddings = mean_pooling(model_output, tokenized_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return {"predictions": embeddings.cpu().tolist()}