from setuptools import setup

setup(
    name="LoLDU",
    version="0.1.0",
    author="Yiming Shi",
    packages=["minloldu"],
    description="A PyTorch implementation of LoLDU",
    license="MIT",
    install_requires=[
        "torch>=2.0.0",
    ],
)
