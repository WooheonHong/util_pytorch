from setuptools import setup

setup(
    name='wooheon_master_lib',
    version='0.0.1',
    decription='wooheon master code pip install'.
    url='https://github.com/WooheonHong/personal_master.git',
    author='Wooheon Hong',
    author_email='quasar103@naver.com',
    packages=['wooheon_master_lib'],
    zip_safe=False,
    install_requires=[
        torchvision==0.8.2
        torch==1.7.1
        numpy==1.19.5
        torchaudio==0.7.0a0+a853dff
    ]
)