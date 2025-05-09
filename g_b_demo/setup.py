from setuptools import setup, find_packages

setup(
    name='eor_l1_demo',
    version='0.1.0',
    packages=find_packages(include=['data', 'data.*', 'models', 'models.*']),
    install_requires=[
        # Minimal dependencies for the modules themselves.
        # PyTorch is expected to be installed via environment.yml or requirements.txt
    ],
    python_requires='>=3.11',
) 