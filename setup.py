from setuptools import find_namespace_packages, setup

setup(
    name="torchbench",
    version="0.1",
    description="Benchmarking library for PyTorch",
    author="PyTorch Team",
    url="https://github.com/pytorch/benchmark",
    # Not every directory with Python code has an __init__.py (for example
    # torchbenchmark/util/framework), and models need their data files
    # (metadata.yaml, patches, ...) at runtime.
    packages=find_namespace_packages(
        include=["torchbenchmark*", "userbenchmark*"], exclude=["*.__pycache__"]
    ),
    package_data={"": ["*"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: BSD 3 License",
        "Programming Language :: Python",
    ],
)
