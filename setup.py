from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wooheon_master_lib",
    version="0.0.1",
    decription="wooheon master code pip install",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WooheonHong/personal_master.git",
    author="Wooheon Hong",
    author_email="quasar103@postech.co.kr",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    license="MIT",
    install_requires=requirements,
)
