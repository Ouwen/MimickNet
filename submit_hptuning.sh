GCS_BUCKET="gs://duke-research-us/mimicknet/experiments/unet_may20"
REGION=us-east1
BUCKET=$GCS_BUCKET

for i in 4 8 16
do
    now=$(date +"%Y%m%d_%H%M%S")
    JOB_NAME="hp_"$i"_"$now""
    JOB_DIR=$BUCKET"/"$JOB_NAME
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        -- \
        --f1 $i \
        --f2 $i \
        --f3 $i \
        --f4 $i \
        --fbn $i \
        --epochs 20
done

for i in 2 4 8
do
    now=$(date +"%Y%m%d_%H%M%S")
    JOB_NAME="pyri_hp_"$i"_"$now""
    JOB_DIR=$BUCKET"/"$JOB_NAME
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        -- \
        --f1  $i \
        --f2  $((i*2)) \
        --f3  $((i*2*2)) \
        --f4  $((i*2*2*2)) \
        --fbn $((i*2*2*2*2)) \
        --epochs 20
done
