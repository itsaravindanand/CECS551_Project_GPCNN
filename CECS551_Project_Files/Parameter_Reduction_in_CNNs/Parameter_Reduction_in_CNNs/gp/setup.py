from setuptools import setup

long_description = '''
GPC NEURAL API is a Keras based neural network API that will allow you to test faster!
This project is uses the state of the art CNN - MobileNet and CIFAR-10 Dataset.
'''

setup(name='gpc',
      version='0.1.1',
      description='Grouped Point-wise Convolution Nerual Network - on MobileNetV1 Architecture ',
      long_description=long_description,
      author='VisionPals',
      url='https://github.com/itsaravindanand/gpc-api.git',
      install_requires=[ # 'tensorflow>=2.10.0', # leaving this as a require sometimes replaces tensorflow
                        'pandas>=0.22.0',
                        'scikit-image>=0.15.0',
                        'opencv-python>=4.1.2.30',
                        'scikit-learn>=0.21.0',
                        'numpy'],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Scientific Research',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=['gpc'])
