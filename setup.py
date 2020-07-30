import setuptools
from distutils.core import setup

# read the contents of README.md
from pathlib import Path

this_directory = Path(__file__).parent
with (this_directory / "readme.md").open(encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="scarpa",
    version="v0.1",
    description="Remove transcranial alternating current artifacts with smooth convolutional adaptive removal of periodic artifacts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Robert Guggenberger",
    author_email="robert.guggenberger@uni-tuebingen.de",
    url="https://github.com/agricolab/scarpa",
    download_url="https://github.com/agricolab/scarpa.git",
    license="MIT",
    packages=setuptools.find_packages(exclude=["test", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Graphics :: Presentation",
        "Topic :: Multimedia :: Sound/Audio :: Players",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
