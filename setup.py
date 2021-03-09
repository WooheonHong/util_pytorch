import io 
from setuptools import find_packages, setup

# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.rst', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

setup(
    name='wooheon_master_lib',
    version='0.0.1',
    decription='wooheon master code pip install',
    long_description=long_description(),
    url='https://github.com/WooheonHong/personal_master.git',
    author='Wooheon Hong',
    author_email='quasar103@naver.com',
    packages=['wooheon_master_lib'],
    zip_safe=False,
    packages=find_packages(),
    license='MIT',
)