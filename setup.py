from distutils.core import setup

setup(
    name='truelearn',
    # major.minor.patch versioning convention
    version='1.1.1',
    packages=['truelearn_experiments', 'truelearn.tests', 'truelearn.learning',
              'truelearn.preprocessing', 'truelearn.learner_models', 'truelearn.metrics',
              'truelearn.datasets', 'truelearn_logging', 'truelearn_utils'],
    url='',
    license='MIT',
    author='',
    author_email='',
    description='',
    install_requires=[
        'numpy>=1.14.1',
        'trueskill>=0.4.5',
        'mpmath>=1.1.0',
    ]
)
