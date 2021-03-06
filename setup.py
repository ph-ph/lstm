from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy==1.14.2',
    'h5py==2.7.1',
    'Keras==2.1.5',
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)

