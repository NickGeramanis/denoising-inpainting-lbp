# Denoising & Inpainting using Loopy Belief Propagation

This package provides an implementation of the loopy belief propagation (LBP)
algorithm in denoising and inpainting greyscale images.

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [License](#license)
- [Authors](#authors)

## Description

This package contains 2 modules:

`image_damager`: Add Gaussian noise to an image and destroy a portion of it.

`denoising_inpainting`: Denoise and inpaint an image using the LBP
algorithm. The current implementation can take many hours for a high number of
iterations or high-resolution images.

The algorithm is based on the following paper:

[Szeliski, R., Zabih, R., Scharstein, D., Veksler, O., Kolmogorov, V., Agarwala, A., Tappen, M., and Rother, C. (2008). A comparative study of energy minimization methods for Markov random fields with smoothness-based priors. IEEE Transactions on Pattern Analysis and Machine Intelligence.](https://ieeexplore.ieee.org/document/4420084)

## Getting Started

### Prerequisites

The following libraries need to be installed:

- NumPy
- OpenCV
- Matplotlib
- PyQt5
  
### Installation

Install the package from the repository with the following commands:

```bash
git clone https://github.com/NickGeramanis/denoising-inpainting-lbp
cd denoising-inpainting-lbp
pip3 install -e .
```

## Usage

To damage an image, execute the function `damage_image()`
of the `image_damager` module with the following parameters:

image_path, noise_mean_value, noise_variance

For example:

```python
from denoising_inpainting_lbp import image_damager

image_damager.damage_image('path/to/image.png', 0, 0.1)
```

Note that a mask image will also be produced that indicates which pixels have
been damaged.

![Image of a boat](/images/boat.png)

![Damaged image](/images/boat-damaged.png)

To perform denoising and inpainting on an image using the LBP algorithm,
execute the function `denoise_inpaint()` of the `denoising_inpainting` module
 with the following parameters:

image_path, mask_image_path, n_iterations, lambda, energy_lower_bound, max_smoothness_penalty

If the smoothness cost function is not truncated do not provide max_smoothness_penalty.

If energy_lower_bound is not known provide 1.

For example:

```python
from denoising_inpainting_lbp import denoising_inpainting

denoising_inpainting.denoise_inpaint('path/to/image.png', 'path/to/mask.png', 1, 5, 37580519.6)
```

![Damaged image of a house](/images/house.png)

![Image after LBP](/images/house-labeled.png)

Furthermore, some unit tests have been implemented in the folder `tests` to verify
the proper functioning of the code.

## Status

Under maintenance.

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)
