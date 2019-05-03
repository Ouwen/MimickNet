now=$(date +"%Y%m%d_%H%M%S")
GCS_BUCKET="gs://duke-research-us/mimicknet/experiments/unet"
BUCKET=$GCS_BUCKET
JOB_NAME="gpu_hptuning_"$now""
JOB_DIR=$BUCKET"/"$JOB_NAME

STAGING_BUCKET=$BUCKET
REGION=us-east1
OUTPUT_PATH=$JOB_DIR

gcloud ai-platform jobs submit training $JOB_NAME \
    --module-name trainer.task \
    --job-dir $JOB_DIR \
    --region $REGION \
    --package-path trainer \
    --config hptuning_config.yaml