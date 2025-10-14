export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # Ensure this region supports v5e TPUs
export REPO_NAME="reranker-repo"
export IMAGE_NAME="tpu-reranker-server"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
export MODEL_DISPLAY_NAME="tpu-reranker-model"
export ENDPOINT_DISPLAY_NAME="tpu-reranker-endpoint"

gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repo for reranker models"

gcloud builds submit . --tag=${IMAGE_URI}

gcloud ai models upload \
    --region=${REGION} \
    --display-name=${MODEL_DISPLAY_NAME} \
    --container-image-uri=${IMAGE_URI} \
    --container-health-route="/health" \
    --container-predict-route="/predict" \
    --container-ports="8080"

# Note down the MODEL_ID from the output of the command above
export MODEL_ID="6409566750236475392"

gcloud ai endpoints create \
    --region=${REGION} \
    --display-name=${ENDPOINT_DISPLAY_NAME}

# Note down the ENDPOINT_ID from the output
export ENDPOINT_ID="8539945295843164160"

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
    --region=${REGION} \
    --model=${MODEL_ID} \
    --display-name="v1-tpu-deployment" \
    --machine-type="ct5lp-hightpu-1t" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --traffic-split=0=100

# Example prediction request
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict" \
    -d '{
          "instances": [
            ["What is the best way to cook a steak?", "For a perfect steak, sear it on a hot pan for 2-3 minutes per side and then finish it in the oven."],
            ["What is the best way to cook a steak?", "A steak is a cut of meat, typically beef."]
          ]
        }'