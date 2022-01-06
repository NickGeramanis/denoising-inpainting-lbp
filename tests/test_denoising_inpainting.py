import pytest

from denoising_inpainting_lbp import denoising_inpainting


def test_image_name():
    with pytest.raises(FileNotFoundError):
        denoising_inpainting.denoise_inpaint('img.png', 'img.png', 1, 1, 1)
