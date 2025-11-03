import os
import subprocess
import time
from threading import Lock
from locust import HttpUser, task, between

# --- Configuration ---
# Set these in your terminal for flexibility, e.g., export ENDPOINT_ID="..."
PROJECT_ID = os.getenv("PROJECT_ID", os.popen("gcloud config get-value project").read().strip())
REGION = os.getenv("REGION", "us-central1")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "8992042486952099840")

# --- Test Data Payloads ---
# A single, organized dictionary for all test cases.
PAYLOADS = {
    "small": {
        1: {"instances": [["What is the capital of France?", "Paris is the capital of France."]]}
    },
    "medium": {
        1: {"instances": [[
            "What were the key contributing factors to the fall of the Roman Empire?", 
            "The fall of the Western Roman Empire was not the result of a single event but a complex, centuries-long process with multiple interrelated causes. Key factors include severe military overstretch, with the empire struggling to defend vast borders against numerous external threats, including Gothic tribes and the Huns, which led to a reliance on unreliable barbarian mercenaries."
        ]]}
    },
    "large": {
        1: {"instances": [[
            "Provide a comprehensive overview of the key principles and mathematical foundations of quantum mechanics, including the Schrödinger equation and the uncertainty principle.", 
            "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science. At its core, the theory departs from classical mechanics by introducing several key principles."
        ]]}
    }
}
# Dynamically create the batch sizes for all payloads
for size in ["small", "medium", "large"]:
    PAYLOADS[size][4] = {"instances": PAYLOADS[size][1]["instances"] * 4}
    PAYLOADS[size][8] = {"instances": PAYLOADS[size][1]["instances"] * 8}


# --- Thread-Safe Token Caching ---
# We will store the token globally and use a lock to ensure only
# one user refreshes it when it expires.
AUTH_TOKEN = None
TOKEN_EXPIRATION_SECONDS = 3000  # 50 minutes, gcloud tokens last 60 mins
TOKEN_FETCH_TIME = 0
token_lock = Lock()

def get_auth_token():
    """Fetches a new auth token using gcloud."""
    print("Refreshing auth token...")
    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True
    ).stdout.strip()
    
    if not token:
        raise Exception("Failed to get gcloud auth token. Is gcloud configured?")
    
    return token

class VertexAIUser(HttpUser):
    """A user that sends reranking requests to a Vertex AI TPU endpoint."""
    
    wait_time = between(0.5, 2.0)
    host = f"https://{REGION}-aiplatform.googleapis.com"
    predict_path = f"/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"

    def on_start(self):
        """
        Authenticates the user when it starts.
        Uses a thread-safe, caching mechanism.
        """
        global AUTH_TOKEN, TOKEN_FETCH_TIME
        
        now = time.time()
        
        # Refresh token if it's None or older than 50 minutes
        if not AUTH_TOKEN or (now - TOKEN_FETCH_TIME > TOKEN_EXPIRATION_SECONDS):
            # Use a lock to ensure only one thread refreshes the token
            with token_lock:
                # Re-check the condition inside the lock in case another
                # thread refreshed it while this one was waiting.
                if not AUTH_TOKEN or (now - TOKEN_FETCH_TIME > TOKEN_EXPIRATION_SECONDS):
                    AUTH_TOKEN = get_auth_token()
                    TOKEN_FETCH_TIME = now
        
        # All users get the same, valid token
        self.client.headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json"
        }

    # --- INSTRUCTIONS ---
    # To run a specific test, UNCOMMENT its "@task" decorator.
    # To disable a test, COMMENT OUT its "@task" decorator.
    # It's best to only have ONE task active at a time for clean results.

    # --- Small Payload Tests ---
    @task
    def test_small_batch_1(self):
        self.client.post(self.predict_path, json=PAYLOADS["small"][1], name="/predict_small_batch_1")

    # @task
    # def test_small_batch_4(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["small"][4], name="/predict_small_batch_4")

    # @task
    # def test_small_batch_8(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["small"][8], name="/predict_small_batch_8")


    # --- Medium Payload Tests ---
    # @task
    # def test_medium_batch_1(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["medium"][1], name="/predict_medium_batch_1")

    # @task
    # def test_medium_batch_4(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["medium"][4], name="/predict_medium_batch_4")
        
    # @task
    # def test_medium_batch_8(self):
    #      self.client.post(self.predict_path, json=PAYLOADS["medium"][8], name="/predict_medium_batch_8")


    # --- Large Payload Tests ---
    # @task
    # def test_large_batch_1(self):
    .client.post(self.predict_path, json=PAYLOADS["large"][1], name="/predict_large_batch_1")

    # @task
    # def test_large_batch_4(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["large"][4], name="/predict_large_batch_4")
        
    # @task
    # def test_large_batch_8(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["large"][8], name="/predict_large_batch_8")