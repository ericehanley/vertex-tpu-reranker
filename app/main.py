import os
from contextlib import asynccontextmanager
from typing import List, Tuple

import torch
import torch_xla.core.xla_model as xm
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# A dictionary to hold the model, a best practice for managing state in FastAPI.
model_handler = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown. The model is loaded and pre-compiled on startup
    to ensure the endpoint is immediately ready for high-performance inference.
    """
    print("🚀 Server starting up...")
    try:
        # 1. Acquire the TPU hardware device
        device = xm.xla_device()
        print("✅ Successfully acquired TPU device.")

        # 2. Load the pre-trained model from Hugging Face
        model_name = 'cross-encoder/ms-marco-MiniLM-L12-v2'
        model = CrossEncoder(model_name, device=device)

        # 3. CRITICAL: Set the model to evaluation mode. This disables layers like
        # Dropout and is essential for deterministic, high-performance inference.
        model.model.eval()

        model_handler["model"] = model
        print(f"✅ Model '{model_name}' loaded and set to eval mode.")

        # --- 4. END-TO-END WARM-UP ROUTINE ---
        # Pre-compiles the model for common batch sizes to eliminate "cold starts".
        # This is the key to achieving consistently low latency.
        print("🔥 Starting end-to-end model warm-up...")
        warmup_batch_sizes = [1, 2, 4, 8]  # Common batch sizes to pre-compile
        dummy_input = [["query", "document"]]

        for batch_size in warmup_batch_sizes:
            print(f"   - Compiling and warming up for batch size: {batch_size}...")
            dummy_batch = dummy_input * batch_size

            # This code block is a perfect mirror of the `predict` function's logic.
            # It ensures both the TPU computation AND the data transfer path are warmed up.
            features = model.tokenizer(dummy_batch, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            features = {key: val.to(model.device) for key, val in features.items()}
            with torch.no_grad():
                outputs = model.model(**features)
                # The .cpu() call is critical to warm up the TPU->CPU data transfer
                _ = outputs.logits.cpu()

        print("✅ Warm-up complete. Endpoint is fully ready for traffic.")

    except Exception as e:
        print(f"⚠️ An error occurred during startup or warm-up: {e}")
        model_handler.clear()

    yield  # The application runs here

    print("👋 Server shutting down...")
    model_handler.clear()


# Initialize the FastAPI app with the lifespan manager
app = FastAPI(title="Cross-Encoder Reranker API", lifespan=lifespan)


class RerankRequest(BaseModel):
    instances: List[Tuple[str, str]]


@app.get(os.environ.get('AIP_HEALTH_ROUTE', '/health'), status_code=200)
def health():
    """Health check endpoint required by Vertex AI."""
    return {"status": "healthy" if "model" in model_handler else "unhealthy"}


@app.post(os.environ.get('AIP_PREDICT_ROUTE', '/predict'))
async def predict(request: RerankRequest):
    """Accepts sentence pairs and returns reranked scores."""
    cross_encoder = model_handler.get("model")
    if not cross_encoder:
        return {"error": "Model not loaded or warm-up failed"}, 503

    # Manually perform the raw inference steps to be compatible with the XLA compiler.
    # 1. Tokenize inputs with static padding to a fixed length.
    features = cross_encoder.tokenizer(
        request.instances,
        padding='max_length',
        truncation=True,
        max_length=512,  # The model's maximum supported sequence length
        return_tensors="pt"
    )

    # 2. Move tokenized data to the TPU
    features = {key: val.to(cross_encoder.device) for key, val in features.items()}

    # 3. Perform inference
    with torch.no_grad():
        outputs = cross_encoder.model(**features)

    # 4. Get results and format the response
    scores = outputs.logits.cpu().tolist()
    # The model outputs a list of lists (e.g., [[score1], [score2]]), so we flatten it.
    predictions = [score[0] for score in scores]

    return {"predictions": predictions}