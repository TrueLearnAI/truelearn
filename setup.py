from distutils.core import setup

setup(
    name='x5gon_data',
    version='0.9.3',
    packages=['test', 'test.transcript_reader', 'test.transcript_reader.data', 'scratch', 'scratch.generate_database',
              'transcript_reader', 'analyses.truelearn_experiments'],
    url='www,x5gon.org',
    license='',
    author='x5gon',
    author_email='m.bulathwela@ucl.ac.uk',
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
