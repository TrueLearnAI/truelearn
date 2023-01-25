from distutils.core import setup

setup(
    name='truelearn',
    # major.minor.patch versioning convention
    version='1.1.1',
    packages=['truelearn'],
    url='',
    license='MIT',
    author='',
    author_email='',
    description='',
    install_requires=[
        'numpy>=1.14.1',
        'trueskill>=0.4.5',
        'mpmath>=1.1.0',
        "scikit-learn>=0.19.1"
    ]
)
