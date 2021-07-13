from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow>=2.0',
    'mitdeeplearning',
    'matplotlib',
    'tensorboard',
    'pydub'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    author='Souranil Sen',
    description='Music Generation RNN'
)