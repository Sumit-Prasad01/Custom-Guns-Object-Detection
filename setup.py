from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name = "Custom Guns Object Detection",
    version = '0.1',
    author = 'Sumit Prasad',
    packages = find_packages(),
    install_requires = requirements,
)