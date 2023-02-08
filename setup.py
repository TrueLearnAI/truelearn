from distutils.core import setup

setup(
    name='truelearn',
    # major.minor.patch versioning convention
    version='1.1.1',
    python_requires='>=3.7',
    packages=['truelearn', 'truelearn.learning', 'truelearn.tests',
              'truelearn.datasets', 'truelearn.models', 'truelearn.utils.metrics',
              'truelearn.utils.visualisations', 'truelearn.utils.persistent',
              'truelearn.preprocessing'],
    url='',
    license='MIT',
    author='',
    author_email='',
    description='',
    long_description=open('README.md').read(),
    install_requires=[
        'trueskill>=0.4.5',
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
