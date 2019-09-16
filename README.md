# MimickNet
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3368028.svg)](https://doi.org/10.5281/zenodo.3368028)

<img src="./examples/cardiac_horizontal_cineloop.gif">

The above is a cineloop of cardiac data with conventional delay-and-sum beamforming and [ReFoCUS](https://ieeexplore.ieee.org/document/8580195) beamforming. We apply clinical-grade post-processing, MimickNet post-processing, and show the unscaled difference between the two.

## Quick Start
You can use the model immediately in [this google colab notebook](https://colab.research.google.com/drive/1VV34wK6Onk_pr5D9fuSdkIeoig6KvK2d)

The model from the paper is provided as `mimicknet.h5`. We also provide a luminance adjusted version with fewer weights which is compatible with matlab `matlab_mimicknet_small_ladj.h5`.

A notebook and sample data is provided under `examples` for use in the following environments:
 - [`python`](https://github.com/Ouwen/MimickNet/blob/master/examples/python3_example.ipynb)
 - [`matlab`](https://github.com/Ouwen/MimickNet/blob/master/examples/matlab_example.m)

[Dr. Mark Palmeri's](https://github.com/mlp6) liver and kidney images are provided as test examples (Thanks Mark!)

## Citing
```
@article{DBLP:journals/corr/abs-1908-05782,
  author    = {Ouwen Huang and
               Will Long and
               Nick Bottenus and
               Gregg E. Trahey and
               Sina Farsiu and
               Mark L. Palmeri},
  title     = {MimickNet, Matching Clinical Post-Processing Under Realistic Black-Box
               Constraints},
  journal   = {CoRR},
  volume    = {abs/1908.05782},
  year      = {2019},
  url       = {http://arxiv.org/abs/1908.05782},
  archivePrefix = {arXiv},
  eprint    = {1908.05782},
  timestamp = {Mon, 19 Aug 2019 13:21:03 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1908-05782},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## TODO
 - Provide phantom dataset to public
 - Mobile Metrics
 - pip package model + util functions
 - Add hyper parameter tuning code for blackbox
 
## Training the model from scratch
The following repo was designed to be run in Google Cloud and makes use of GCS for logging.
```
python3 -m trainer.blackbox_task
```
Different hyperparameters can be selected. Different tasks are shown in the root directory of the `trainer`. `blackbox_task.py`, and `graybox_task.py` are for running the blackbox and graybox tasks in the paper: "MimickNet, Matching Clinical Post-Processing Under Realistic Black-Box Constraints". `blackbox_paper_v1.py`, is the exact code run for the first release.

### Custom data
The current model uses the dataset loader found in `./trainer/utils/dataset.py`. This dataset takes in two lists from different domains `A` and `B`. The dataset returns elements which are 4D tensors. Each dimension represents `[batch, height, width, 1]`. You can create your own dataset class, it simply needs to return a 4D tensor, and count the total number of elements. See [here](https://www.tensorflow.org/datasets/overview) on how to use tensorflow datasets.

# Contributing and Issues
Contributions are welcome in the form of pull requests.
For issues, please provide a minimal example of code which results in an error.

# License
Note that the model provided [here](https://github.com/Ouwen/MimickNet/tree/master/examples) is under a different license than the software code.

## Model License
CC 4.0 Attribution-NonCommercial International

## Software License
Copyright 2019 Ouwen Huang

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
