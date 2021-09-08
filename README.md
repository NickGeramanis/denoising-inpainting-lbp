# Denoising & Inpainting using Loopy Belief Propagation

This project provides an implementation of the loopy belief propagation (LBP)
algorithm in denoising and inpainting greyscale images.

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Status](#status)
- [License](#license)
- [Authors](#authors)

## Description

This project contains 2 scripts:

`damage_image.py`: Add Gaussian noise to an image and destroy a portion of it.

`denoising_inpainting.py`: Denoise and inpaint an image using the LBP
algorithm. The current implementation can take many hours for a high number of
iterations or high resolution images.

The algorithm is based on the following paper:

Szeliski, R., Zabih, R., Scharstein, D., Veksler, O., Kolmogorov, V., Agarwala,
A., Tappen, M., and Rother, C. (2008). A comparative study of energy
minimization methods for Markov random fields with smoothness-based priors.
IEEE Transactions on Pattern Analysis and Machine Intelligence.

## Getting Started

### Prerequisites

The following libraries need to be installed:

- NumPy
- OpenCV
- Matplotlib

## Usage

To create a damaged image (and its mask), execute the script `damage_image.py`
with the following parameters:

image_name, gaussian_noise_mean_value, gaussian_noise_variance

for example:

```bash
python3 damage_image.py boat.png 0 0.1
```

![Image of a boat](/images/boat.png)

![Produced damaged image](/images/boat-damaged.png)

To perform denoising and inpainting on an image, execute the
script `denoising_inpainting.py` with the following parameters:

image_name, number_of_iterations, lambda, maximum_smoothness_penalty,
energy_lower_bound

Note that the mask image must also be in the given path

If the smoothness cost function is not truncated, provide inf as
maximum_smoothness_penalty.

If energy_lower_bound is not known provide 1.

for example:

```bash
python3 denoising_inpainting.py house.png 5 5 inf 37580519.6
```

![Damaged image of a house](/images/house-damaged.png)

![Image after LBP](/images/house-labeled.png)

Furthermore, some unit tests have been implemented in folder `tests` to verify
the proper functioning of the code.

## Status

Under maintenance.

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

