from setuptools import setup, find_packages

setup(
    name='torchbench',
    version='0.1',
    description='Benchmarking library for PyTorch',
    author='PyTorch Team',
    url='https://github.com/pytorch/benchmark',  # replace with the actual URL of your project
    packages=find_packages(include=['torchbenchmark*', 'userbenchmark*']),
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD 3 License',  # assuming MIT License, modify accordingly
        'Programming Language :: Python',
    ],
)
