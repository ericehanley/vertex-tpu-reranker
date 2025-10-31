# **IMPORTANT** Checks before running the below commands:
# 1.) Update Dockerfile with desired directory, i.e. app_embedding or app_encoding
# i.e., app_embedding for mxbai or roberta and app_encoding for mini
# 2.) If deploying app_embedding ensure the correct model is selected in main.py
# lines 15 through 19, i.e. mixedbread or roberta
# 3.) Update below variables accordingly to reflect which model will be deployed
# 4.) If executing locustfile, update ENDPOINT_ID on line 9 of locustfile.py

export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # ensure this region supports v5e TPUs
export REPO_NAME="tpu-roberta-v1-repo" # update accordingly
export IMAGE_NAME="tpu-roberta-v1-server" # update accordingly
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
export MODEL_DISPLAY_NAME="tpu-roberta-v1-model" # update accordingly
export ENDPOINT_DISPLAY_NAME="tpu-roberta-v1-endpoint" # update accordingly

# create artifact repository
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repo for reranker models"

# build image
gcloud builds submit . --tag=${IMAGE_URI}

# upload model to vertex
gcloud ai models upload \
    --region=${REGION} \
    --display-name=${MODEL_DISPLAY_NAME} \
    --container-image-uri=${IMAGE_URI} \
    --container-health-route="/health" \
    --container-predict-route="/predict" \
    --container-ports="8080"

# note down the MODEL_ID from the output of the command above
# or Vertex UI and enter here
export MODEL_ID="3229497647731572736"

# create endpoint
gcloud ai endpoints create \
    --region=${REGION} \
    --display-name=${ENDPOINT_DISPLAY_NAME}

# note down the ENDPOINT_ID from the output of the command above
# or Vertex UI and enter here
export ENDPOINT_ID="7175965937214947328"

# deploy model to endpoint
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
    --region=${REGION} \
    --model=${MODEL_ID} \
    --display-name="v1-tpu-deployment" \
    --machine-type="ct5lp-hightpu-1t" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --traffic-split=0=100

# example prediction request for encoder
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

# example prediction request for encoder
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