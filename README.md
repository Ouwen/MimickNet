# MimickNet

![cardiac cineloop](./examples/cardiac_cineloop.gif | width=200)
From top to bottom, the above is a cineloop of the heart with no post-processing, clinical-grade post-processing, MimickNet post-processing, and the absolute difference between MimickNet and clinical-grade post-processing. Histogram Matching is applied to raw beamformed data.

### Using the model
The model from the paper is provided as `mimicknet.h5`. We also provide a luminance adjusted version `mimicknet_luminance_adjusted.h5`.
A notebook and sample data is provided under `examples` for use in the following environments:
 - python
 - matlab

### Training the model
The following repo was designed to be run in Google Cloud and makes use of GCS for logging.
```
python3 -m trainer.blackbox_paper
```
Different hyperparameters can be selected. Different tasks are shown in the root directory of the `trainer`. `blackbox_paper.py`, and `graybox_paper.py` are the configurations used in the paper: "MimickNet, Matching Clinical Post-Processing Under Realistic Black-Box Constraints". `blackbox_simple.py` strips away many of the options used for hyperparameter exploration for readability. If you have your own dataset it is recommended to start from this file.

#### Custom data
The current model uses the dataset loader found in `trainer.utils.dataset`. This dataset takes in two lists from different domains `A` and `B`. The dataset returns elements which are 4D tensors. Each dimension represents `[batch, height, width, 1]`. You can create your own dataset class, it simply needs to return a 4D tensor, and the total count of elements.