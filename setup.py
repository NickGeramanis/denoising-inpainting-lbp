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
      python_requires='==3.13.2',
      install_requires=['opencv-python==4.11.0.86',
                        'numpy==2.2.4',
                        'matplotlib==3.10.1'])
