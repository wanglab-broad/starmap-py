import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="starmap",
    version="0.1.8",
    author="Jiahao Huang",
    author_email="jiahao@broadinstitute.org",
    description="A python package of STARMap bioinformatis data analysis pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jiahaoh/starmap_clean",
    packages=['starmap'],
    # install_requires=[
    #     l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    # ],
    # packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)