import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hmm-replay-seblee97",
    version="0.0.1",
    author="Sebastian Lee",
    author_email="sebastianlee.1997@yahoo.co.uk",
    description="HMM implementation for multi-teachers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seblee97/hmm-replay",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
