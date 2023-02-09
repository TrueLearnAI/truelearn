from distutils.core import setup
from setuptools import find_packages
import truelearn

setup(
    name='truelearn',
    version=truelearn.__version__,
    python_requires='>=3.7',
    packages=find_packages("truelearn"),
    url='',
    license='MIT',
    author='KD-7,sahanbull,yuxqiu,deniselezi,aaneelshalman',
    author_email='',
    description='',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy>=1.14.1',
        'trueskill>=0.4.5',
        'mpmath>=1.1.0',
        "scikit-learn>=0.19.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ]
)
if __name__ == "__main__":
    setup()
