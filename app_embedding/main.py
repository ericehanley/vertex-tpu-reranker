import os
from contextlib import asynccontextmanager
from typing import List

import torch
import torch_xla.core.xla_model as xm
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# A dictionary to hold the model components.
model_handler = {}

# --- CHOOSE YOUR MODEL ---
# To deploy all-distilroberta-v1, use this line:
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'

# To deploy mxbai-embed-large-v1, use this line instead:
# MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup: load, optimize, compile, and warm up the model.
    """
    print(f"🚀 Server starting up with model: {MODEL_NAME}")
    try:
        # 1. Acquire the TPU hardware device
        device = xm.xla_device()
        print(f"Successfully acquired TPU device: {device}")

        # 2. Load the pre-trained model
        # The mxbai model requires trusting remote code from the repo.
        trust_remote = 'mxbai' in MODEL_NAME
        model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=trust_remote)

        # 3. CRITICAL: Set to evaluation mode for inference
        model.eval()
        print(f"Model loaded in float32 precision and set to eval mode.")

        # --- 4. PERFORMANCE: Compile the model with the torch_xla backend ---
        print("Compiling the model for the XLA backend...")
        # SentenceTransformer is an nn.Module, so we can compile it directly
        compiled_model = torch.compile(model, backend="torch_xla")
        model_handler["compiled_model"] = compiled_model
        print("Model compilation complete.")

        # --- 5. WARM-UP ROUTINE ---
        # Warms up the compiled model for common batch sizes to avoid cold starts.
        print("Starting end-to-end model warm-up...")
        # Using a wider range of batch sizes for robust warm-up
        warmup_batch_sizes = [1, 2, 4, 8, 16, 32] 
        dummy_input = ["This is a warmup sentence."]

        for batch_size in warmup_batch_sizes:
            print(f"   - Warming up for batch size: {batch_size}...")
            with torch.no_grad():
                # The .encode() method is the high-level API for SentenceTransformer
                embeddings = compiled_model.encode(
                    dummy_input * batch_size, 
                    convert_to_tensor=True
                )
            # Synchronize by moving a small slice back to CPU
            _ = embeddings[0].cpu()

        print("✅ Warm-up complete. Endpoint is fully ready for traffic.")

    except Exception as e:
        print(f"❌ An error occurred during startup or warm-up: {e}")
        model_handler.clear()

    yield  # The application runs here

    print("🔻 Server shutting down...")
    model_handler.clear()


# Initialize the FastAPI app
app = FastAPI(title="Sentence Embedding API", lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    instances: List[str] = Field(..., example=["This is a test sentence."])

class EmbeddingResponse(BaseModel):
    predictions: List[List[float]]


@app.get(os.environ.get('AIP_HEALTH_ROUTE', '/health'), status_code=200)
def health():
    """Health check endpoint required by Vertex AI."""
    return {"status": "healthy" if "compiled_model" in model_handler else "unhealthy"}


@app.post(
    os.environ.get('AIP_PREDICT_ROUTE', '/predict'),
    response_model=EmbeddingResponse
)
async def predict(request: EmbeddingRequest):
    """Generates embeddings for a list of input sentences."""
    compiled_model = model_handler.get("compiled_model")
    if not compiled_model:
        return {"error": "Model not loaded or warm-up failed"}, 503

    with torch.no_grad():
        # The .encode() method handles tokenization, inference, and pooling.
        embeddings = compiled_model.encode(
            request.instances,
            # convert_to_numpy is efficient for the final list conversion
            convert_to_numpy=True 
        )

    return {"predictions": embeddings.tolist()}