GCS_BUCKET="gs://duke-research-us/mimicknet/experiments/unet"
REGION=us-east1
BUCKET=$GCS_BUCKET

for i in 1 2 4 8 16
do
    now=$(date +"%Y%m%d_%H%M%S")
    JOB_DIR=$BUCKET"/ssim_hp_"$i"_"$now""
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        --f1 $i \
        --f2 $i \
        --f3 $i \
        --f4 $i \
        --fbn $i \
        --epochs 10 \
        --l_ssim 1

    now=$(date +"%Y%m%d_%H%M%S")
    JOB_DIR=$BUCKET"/mse_hp_"$i"_"$now""
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        --f1 $i \
        --f2 $i \
        --f3 $i \
        --f4 $i \
        --fbn $i \
        --epochs 10 \
        --l_mse 1 

    now=$(date +"%Y%m%d_%H%M%S")
    JOB_DIR=$BUCKET"/mae_hp_"$i"_"$now""
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        --f1 $i \
        --f2 $i \
        --f3 $i \
        --f4 $i \
        --fbn $i \
        --epochs 10 \
        --l_mae 1
done

for i in 1 2 4 8
do
    now=$(date +"%Y%m%d_%H%M%S")
    JOB_DIR=$BUCKET"/ssim_pyri_hp_"$i"_"$now""
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        --f1  $i \
        --f2  $i*2 \
        --f3  $i*2*2 \
        --f4  $i*2*2*2 \
        --fbn $i*2*2*2*2 \
        --epochs 10 \
        --l_ssim 1

    now=$(date +"%Y%m%d_%H%M%S")
    JOB_DIR=$BUCKET"/mse_pyri_hp_"$i"_"$now""
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        --f1  $i \
        --f2  $i*2 \
        --f3  $i*2*2 \
        --f4  $i*2*2*2 \
        --fbn $i*2*2*2*2 \
        --epochs 10 \
        --l_mse 1 

    now=$(date +"%Y%m%d_%H%M%S")
    JOB_DIR=$BUCKET"/mae_pyri_hp_"$i"_"$now""
    gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name trainer.task \
        --job-dir $JOB_DIR \
        --region $REGION \
        --package-path trainer \
        --python-version 3.5 \
        --config hptuning_config.yaml \
        --f1  $i \
        --f2  $i*2 \
        --f3  $i*2*2 \
        --f4  $i*2*2*2 \
        --fbn $i*2*2*2*2 \
        --epochs 10 \
        --l_mae 1
done
