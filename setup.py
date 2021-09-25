from setuptools import setup

setup(name='denoising_inpainting_lbp',
      version='0.1.0',
      packages=['denoising_inpainting_lbp'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description='Loopy Belief Propagation algorithm for denoising and inpainting greyscale images',
      url='https://github.com/NickGeramanis/denoising-inpainting-lbp',
      license='GPLV3',
      install_requires=['opencv-python', 'numpy', 'matplotlib'])
