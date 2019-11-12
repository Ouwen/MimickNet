# Figures
Figures used for publication. 
Code to generate image figures are avalible in [this colab notebook](https://colab.research.google.com/drive/10A2-hNHqPNNfqETlx_RzK0eknjgvEE0d).

Code to generate chart figures are avalible in [this colab notebook](https://colab.research.google.com/drive/1stCxM5jj0mrdL6alNT1WqLrBTK7b6qTO).

# Supplementary Table
In the theoretical gray-box case where before and after
paired images are available, we explore different possible Unet
encoder-decoder hyperparameters. For each hyperparameter
variation, we trained a triplet of models that optimize for
SSIM, MSE, and MAE. We note that within each triplet,
models using the SSIM minimization objective have the best
SSIM and PSNR. We are primarily interested in the best
SSIM metric since it was originally formulated to model
the human visual system (Bovik 2004). In the table below, the best average
metrics of each column are in bold. Many of the metrics across
model variations are not significantly different, but the SSIM
for every model is above 0.967. For subsequent worst-case
performance analysis, we used the 52993 parameter model
optimized on SSIM loss. This model corresponds to the same
generator structure used in Fig. 2 except with a 3×3 instead
of a 7×3 filter.

| Loss | Params | MSE 10^-3 | MAE 10^-2 | PSNR | SSIM |
| ---- | ------ | --------- | --------- | ---- | ---- |
| ssim | 13377  | 2.78±3.22 | 3.97±2.40 | 27.4±3.9 | 0.967±0.015 |
| mse | 13377 | 2.40±2.65 | 3.76±2.00 | 27.7±3.4 | 0.947±0.022 |
| mae | 13377 | 2.51±2.86 | 3.83±2.13 | 27.6±3.5 | 0.946±0.018 |
| ssim | 29601 | 2.63±3.10 | 3.91±2.40 | 27.9±4.0 | 0.967±0.015 |
| mse | 29601 | 2.19±2.25 | 3.61±1.81 | 27.9±3.2 | 0.940±0.019 |
| mae | 29601 | 3.46±3.20 | 4.58±2.16 | 25.7±3.0 | 0.895±0.028 |
| ssim | 34849 | 2.49±2.88 | 3.78±2.28 | 27.9±3.9 | 0.975±0.013 |
| mse | 34849 | 2.27±2.41 | 3.67±1.92 | 27.9±3.3 | 0.950±0.019 |
| mae | 34849 | 2.31±2.54 | 3.68±1.96 | 27.9±3.4 | 0.951±0.016 |
| ssim | 52993 | 2.28±2.77 | 3.65±2.24 | 28.5±4.2 | 0.979±0.013 |
| mse | 52993 | 2.19±2.40 | 3.60±1.92 | 28.1±3.4 | 0.956±0.017 |
| mae | 52993 | 2.11±2.35 | 3.52±1.89 | 28.3±3.4 | 0.959±0.015 |
| ssim | 77185 | 2.38±2.91 | 3.70±2.28 | 28.3±4.0 | 0.976±0.015 |
| mse | 77185 | 2.02±2.09 | 3.46±1.70 | 28.3±3.2 | 0.946±0.022 |
| mae | 77185 | 2.14±2.23 | 3.55±1.80 | 28.0±3.2 | 0.947±0.020 |
| ssim | 117697 | 2.22±2.65 | 3.59±2.11 | 28.4±3.9 | 0.977±0.014 |
| mse | 117697 | 2.72±2.51 | 4.07±1.95 | 26.9±3.1 | 0.931±0.023 |
| mae | 117697 | 2.93±2.93 | 4.18±2.11 | 26.7±3.3 | 0.927±0.022 |
| ssim | 330401 | 2.25±2.79 | 3.61±2.22 | 28.6±4.1 | 0.977±0.013 |
| mse | 330401 | 2.15±2.20 | 3.58±1.83 | 28.1±3.4 | 0.958±0.016 |
| mae | 330401 | 2.23±2.42 | 3.61±1.89 | 28.0±3.4 | 0.958±0.016 |
| ssim | 733025 | 2.63±3.06 | 3.93±2.33 | 27.7±4.0 | 0.967±0.015 |
| mse | 733025 | 2.40±2.51 | 3.79±1.97 | 27.7±3.4 | 0.945±0.023 |
| mae | 733025 | 2.80±2.83 | 4.09±2.04 | 26.9±3.2 | 0.927±0.022 |

Without leakyReLU activation, we have difficulty achieving convergence with models using fewer than 117697 params. We show performance with different cycle-consistency loss and increasing parameters. The row labeled “ver”, is a model trained only on Verasonics Vantage data with MAE optimization.

| Loss | Params | MSE 10^-3 | MAE 10^-2 | PSNR | SSIM |
| ---- | ------ | --------- | --------- | ---- | ---- |
|ssim  | 117697 | 7.26±10.5 | 6.54±4.38 | 23.9±4.41 | 0.883±0.091 |
|mse | 117697 | 6.83±11.1 | 6.31±4.39 | 24.7±4.95 | 0.930±0.089 |
|mae | 117697 | 6.79±9.89 | 6.30±4.27 | 24.4±4.67 | 0.900±0.085 |
|ssim | 7.76M | 4.45±5.71 | 5.14±3.12 | 25.6±4.08 | 0.918±0.078 |
|mse | 7.76M | 6.23±6.30 | 6.14±3.24 | 23.6±3.57 | 0.897±0.052 | 
|mae | 7.76M | 6.20±9.10 | 6.02±4.21 | 25.1±5.05 | 0.918±0.084 | 
|ver  | 7.76M | 6.13±8.95 | 5.99±4.19 | 25.2±5.08 | 0.916±0.083 |


