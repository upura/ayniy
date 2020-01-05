import os
from setuptools import find_packages, setup


with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()


def get_version():
    # type: () -> str

    version_filepath = os.path.join(os.path.dirname(__file__), 'ayniy', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]
    assert False


def get_install_requires():
    install_requires = [
        'tqdm',
        'numpy',
        'scikit-learn',
        'pandas',
        'joblib',
        'kaggler'
    ]
    return install_requires


def get_extra_requires():
    extras = {
        'test': ['pytest', 'ipython', 'jupyter', 'notebook', 'tornado==5.1.1'],
        'document': ['sphinx', 'sphinx_rtd_theme']
    }
    return extras


setup(
    name='ayniy',
    version=get_version(),
    author='upura',
    packages=find_packages(),
    include_package_data=True,
    license='BSD License',
    description='Ayniy is a machine learning data pipeline especially for Kaggle.',
    long_description=README,
    url='https://github.com/upura',
    author_email='upura0@gmail.com',
    install_requires=get_install_requires(),
    tests_require=get_extra_requires()['test'],
    extras_require=get_extra_requires(),
    classifiers=[
        'Environment :: CLI Environment',
        'Framework :: scikit-learn',
        'Framework :: scikit-learn :: X.Y',  # replace "X.Y" as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
