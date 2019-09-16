GCS_BUCKET="gs://duke-research-us/mimicknet/ai_platform/experiments"
IMAGE_URI=gcr.io/duke-ultrasound/mimicknet:tf2rc0
REGION=us-east1

docker build -f Dockerfile -t $IMAGE_URI .
docker push $IMAGE_URI

JOB_NAME=MimickNet_Blackbox_Phantom$(date +%Y_%m_%d_%H%M%S)
JOB_DIR=$GCS_BUCKET"/"$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
  --master-image-uri $IMAGE_URI \
  --scale-tier custom \
  --master-machine-type standard_p100 \
  --region $REGION \
  --job-dir $JOB_DIR \
  -- \
  --train_das_csv 'gs://duke-research-us/mimicknet/data/training-v2-verasonics-phantom.csv' \
  --train_clinical_csv 'gs://duke-research-us/mimicknet/data/training-v2-clinical-phantom.csv'
gcloud ai-platform jobs describe $JOB_NAME

JOB_NAME=MimickNet_Blackbox_VeraClin$(date +%Y_%m_%d_%H%M%S)
JOB_DIR=$GCS_BUCKET"/"$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
  --master-image-uri $IMAGE_URI \
  --scale-tier custom \
  --master-machine-type standard_p100 \
  --region $REGION \
  --job-dir $JOB_DIR \
  -- \
  --train_das_csv 'gs://duke-research-us/mimicknet/data/training-v2-verasonics.csv' \
  --train_clinical_csv 'gs://duke-research-us/mimicknet/data/training-v2-clinical.csv'
gcloud ai-platform jobs describe $JOB_NAME
