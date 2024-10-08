from setuptools import setup

setup(name='denoising_inpainting_lbp',
      version='1.0.0',
      packages=['denoising_inpainting_lbp'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description=('Loopy Belief Propagation algorithm '
                   'for denoising and inpainting greyscale images'),
      url='https://github.com/NickGeramanis/denoising-inpainting-lbp',
      license='GPLV3',
      python_requires='==3.12.6',
      install_requires=['opencv-python==4.10.0.84',
                        'numpy==2.1.2',
                        'matplotlib==3.9.2'])
