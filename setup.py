import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="starmap-py",
    version="0.1.0",
    author="Jiahao Huang",
    author_email="jiahaoh@mit.edu",
    description="The data analysis tool kit for spatial transcriptomics data generated by STARmap-related methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wanglab-broad/starmap-py",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),
    # install_requires=[
    #     l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    # ],
    # packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: BSD 3-Clause License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.9',
)