# MimickNet

![Alt Text](cardiac_cineloop.gif)
From top to bottom, the above is a cineloop of the heart with no post-processing, clinical-grade post-processing, MimickNet post-processing, and the absolute difference between MimickNet and clinical-grade post-processing. 

### Training the model
The following repo was designed to be run in Google Cloud and makes use of GCS for logging.

```
python3 -m trainer.cyclegan_paper
```

### Using the model
The model from the paper is provided as `mimicknet.h5`. We also provide a luminance adjusted version `mimicknet_luminance_adjusted.h5`. A hyperparameter tuned version is on the way.

