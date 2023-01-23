from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.2"
DESCRIPTION = "Simple path editing"

# Setting up
setup(
    name="pathprocessing",
    version=VERSION,
    author="wrangelvid (David von Wrangel)",
    author_email="<wrangelvid@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(include=["pathprocessing"]),
    install_requires=[
        "numpy >= 1.21.5",
        "svgpathtools >= 1.5.1",
        "rdp >= 0.8",
        "matplotlib >= 3.5.1",
        "pycairo >= 1.20.1",
        "qrcode >= 7.3.1",
        "pillow >= 9.0.1",
        "torch >= 1.13.1",
        "torchvision >= 0.14.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    license="MIT",
)
