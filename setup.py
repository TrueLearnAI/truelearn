from distutils.core import setup

setup(
    name='truelearn',
    # major.minor.patch versioning convention
    version='1.1.1',
    packages=['truelearn_experiments', 'truelearn.unit_tests', 'truelearn.bayesian_models',
              'truelearn.preprocessing', 'truelearn.visualisations', 'truelearn.visualisations.plotting_utils'],
    url='',
    license='MIT',
    author='',
    author_email='',
    description='',
    install_requires=[
        'numpy>=1.14.1',
        'pandas>=0.22.0',
        'scipy>=1.0.1',
        'nltk>=3.2.5',
        'xmltodict>=0.11.0',
        'ujson>=1.35',
        'scikit-learn>=0.19.1',
        'joblib>=0.14.1',
        'trueskill>=0.4.5',
        'mpmath>=1.1.0',
        'networkx>=2.5',
        'statsmodels>=0.12.1',
        'wordcloud'
    ]
)
