from denoising_inpainting_lbp import image_damager
from denoising_inpainting_lbp import denoising_inpainting

image_damager.damage_image('images/boat.png', 0, 0.1)
denoising_inpainting.denoise_inpaint('images/boat-damaged.png',
                                     'images/boat-mask.png',
                                     1,
                                     5,
                                     37580519.6)
