from denoising_inpainting_lbp import image_damager
from denoising_inpainting_lbp import denoising_inpainting

image_damager.damage_image('images/boat.pnp', 0, 0.1)
denoising_inpainting.denoise_inpaint('images/house-damaged.png',
                                     'images/house-mask.png',
                                     1,
                                     5,
                                     37580519.6)
