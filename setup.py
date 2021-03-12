# python

from setuptools import setup


setup(
    name="InputTensorFI",
    packages=["InputTensorFI"],
    version="1.0.0",
    license="MIT",
    description="Dynamic Injection in the input layer of tensorflow.",
    author="Marc NGUYEN",
    author_email="nguyen_marc@live.fr",
    url="https://github.com/Darkness4/InputTensorFI",
    install_requires=[
        "matplotlib>=3,<4",
        "tensorflow>=2,<3",
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
