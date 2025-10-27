import os
import subprocess
from locust import HttpUser, task, between

# --- Configuration ---
# Set these in your terminal for flexibility, e.g., export ENDPOINT_ID="..."
PROJECT_ID = os.getenv("PROJECT_ID", os.popen("gcloud config get-value project").read().strip())
REGION = os.getenv("REGION", "us-central1")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "8539945295843164160")

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


class VertexAIUser(HttpUser):
    """A user that sends reranking requests to a Vertex AI TPU endpoint."""
    
    wait_time = between(0.5, 2.0)
    host = f"https://{REGION}-aiplatform.googleapis.com"
    predict_path = f"/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"

    def on_start(self):
        """Authenticates the user when it starts."""
        token = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True
        ).stdout.strip()
        
        self.client.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    # --- INSTRUCTIONS ---
    # To run a specific test, UNCOMMENT its "@task" decorator.
    # To disable a test, COMMENT OUT its "@task" decorator.
    # It's best to only have ONE task active at a time for clean results.

    # --- Small Payload Tests ---
    # @task
    # def test_small_batch_1(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["small"][1], name="/predict_small_batch_1")

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
    #     self.client.post(self.predict_path, json=PAYLOADS["large"][1], name="/predict_large_batch_1")

    # @task
    # def test_large_batch_4(self):
    #     self.client.post(self.predict_path, json=PAYLOADS["large"][4], name="/predict_large_batch_4")
        
    @task
    def test_large_batch_8(self):
        self.client.post(self.predict_path, json=PAYLOADS["large"][8], name="/predict_large_batch_8")