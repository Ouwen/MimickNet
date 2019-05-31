GCS_BUCKET="gs://duke-research-us/mimicknet/experiments/unet_may31_clipping"
REGION=us-east1
BUCKET=$GCS_BUCKET

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="MimickNet_clipped_full_bayes_search_"$i"_"$now""
JOB_DIR=$BUCKET"/"$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
    --module-name trainer.task \
    --job-dir $JOB_DIR \
    --region $REGION \
    --package-path trainer \
    --python-version 3.5 \
    --config hptuning_config.yaml \
    -- \
    --epochs 100
    