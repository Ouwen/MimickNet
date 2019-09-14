# PROJECT_ID: your project's id. Use the PROJECT_ID that matches your Google Cloud Platform project.
export PROJECT_ID=duke-ultrasound

# BUCKET_ID: the bucket id you created above.
export BUCKET_ID=duke-research-us

# IMAGE_REPO_NAME: where the image will be stored on Cloud Container Registry
export IMAGE_REPO_NAME=mimicknet

# IMAGE_TAG: an easily identifiable tag for your docker image
export IMAGE_TAG=tf2rc0

# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
export REGION=us-central1

# JOB_NAME: the name of your job running on AI Platform.
export JOB_NAME=mimicknet_tf2$(date +%Y%m%d_%H%M%S)

docker build -f Dockerfile -t $IMAGE_URI .

docker run --gpus all $IMAGE_URI