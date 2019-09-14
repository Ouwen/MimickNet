# PROJECT_ID: your project's id. Use the PROJECT_ID that matches your Google Cloud Platform project.
export PROJECT_ID=duke-ultrasound

# BUCKET_ID: the bucket id you created above.
export BUCKET_ID=duke-research-us
export GCS_BUCKET="gs://duke-research-us/mimicknet/ai_platform/experiments"


# IMAGE_REPO_NAME: where the image will be stored on Cloud Container Registry
export IMAGE_REPO_NAME=mimicknet

# IMAGE_TAG: an easily identifiable tag for your docker image
export IMAGE_TAG=tf2rc0

# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
export REGION=us-east1

# JOB_NAME: the name of your job running on AI Platform.
export JOB_NAME=MimickNet_Blackbox_$(date +%Y_%m_%d_%H%M%S)
export JOB_DIR=$GCS_BUCKET"/"$JOB_NAME

docker build -f Dockerfile -t $IMAGE_URI .
docker push $IMAGE_URI

gcloud ai-platform jobs submit training $JOB_NAME \
  --master-image-uri $IMAGE_URI \
  --scale-tier custom \
  --master-machine-type standard_p100 \
  --region $REGION \
  --job-dir $JOB_DIR \
  -- \
  --kernel_height 7

gcloud ai-platform jobs describe $JOB_NAME

