from setuptools import setup

setup(name='denoising_inpainting_lbp',
      version='0.1.0',
      packages=['denoising_inpainting_lbp'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description=('Loopy Belief Propagation algorithm '
                   'for denoising and inpainting greyscale images'),
      url='https://github.com/NickGeramanis/denoising-inpainting-lbp',
      license='GPLV3',
      python_requires='==3.10.10',
      install_requires=['opencv-python==4.7.0.72',
                        'numpy==1.24.2',
                        'matplotlib==3.7.1',
                        'pyqt5==5.15.9'])
