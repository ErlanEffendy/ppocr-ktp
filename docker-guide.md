# Docker Guide for PPOCR KTP API

This guide provides instructions on how to run the PPOCR KTP and NPWP extraction API using Docker.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Running the Application

1. **Build and start the container:**
   Run the following command in the root of the project:
   ```bash
   docker-compose up -d --build
   ```

2. **Check the logs:**
   If you want to view the logs to ensure everything started correctly, use:
   ```bash
   docker-compose logs -f
   ```

3. **Access the API:**
   The API will be available at `http://localhost:8001`.
   - **Swagger UI documentation:** `http://localhost:8001/docs`
   - **ReDoc documentation:** `http://localhost:8001/redoc`

## Data Persistence (Volumes)
The `docker-compose.yml` is configured with volumes so data is not lost when the container is stopped:
- `./uploaded_files`: Persists uploaded KTP and NPWP images on your local machine.
- `./.cache_regional`: Persists cached regional data (Provinces, Cities, etc.) from the regional API.

## Stopping the Application
To stop the running container, execute:
```bash
docker-compose down
```

## Hardware Acceleration (CPU vs GPU)
By default, the `Dockerfile` is configured to pre-install the **CPU-only** version of PyTorch. This dramatically reduces the Docker build time and image size (saving ~2.5GB) since the API does not currently require a GPU.

**If you want to use a GPU in the future:**
1. Ensure your host machine has Nvidia drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
2. Open the `Dockerfile` and **delete** this line to allow `ultralytics` to install the default GPU-enabled PyTorch:
   ```dockerfile
   RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
3. Open `docker-compose.yml` and add the GPU device reservation:
   ```yaml
   services:
     ppocr-api:
       # ... existing config ...
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```
4. Rebuild the container: `docker-compose up -d --build`
