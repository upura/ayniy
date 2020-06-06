from codecs import open
from os import path

from setuptools import find_packages, setup


def get_long_description():
    here = path.abspath(path.dirname(__file__))

    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def get_version():
    version_filepath = path.join(path.dirname(__file__), 'nyaggle', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]


setup(
    name='Ayniy',
    packages=find_packages(),

    version=get_version(),

    license='MIT',

    install_requires=[
        'git+https://github.com/stanfordmlgroup/ngboost.git'
        'ginza',
        'japanize-matplotlib',
        'matplotlib',
        'numpy',
        'pandas',
        'optuna>=1.0.0',
        'seaborn',
        'sklearn',
        'kaggler',
        'mecab-python3',
        'mlflow',
        'neologdn',
        'tqdm',
        'transformers==2.5.1',
    ],

    extras_require={
        'tests': ['pytest>=3.6', 'pytest-remotedata>=0.3.1']
    },

    author='upura',
    author_email='upura0@gmail.com',
    url='https://github.com/upura/ayniy',
    description='Ayniy is a supporting tool for machine learning competitions.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    keywords='ayniy kaggle',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ]
)
