# Use a standard Python base image
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
# Set the working directory inside the container
WORKDIR /app_embedding

# Copy the requirements file and install dependencies
COPY ./app_embedding/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app_embedding/ .

# Expose the port the server will run on
EXPOSE 8080

# Command to run the Uvicorn server
# This will be the entry point for the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]