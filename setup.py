from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

setup(
    name='paraphraser',
    version='0.1.0a1',
    description='Generate sentence paraphrases given an input sentence',
    long_description=long_desc,
    url='https://github.com/vsuthichai/paraphraser',
    author='Victor Suthichai',
    author_email='victor.suthichai@gmail.com',

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],

    keywords=[
        'paraphraser'
    ],

    py_modules=['paraphraser.synonym_model'],
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['nltk', 'spacy', 'ipython'],
    extras_require={

    },
    package_data={

    },
    data_files=[],
    entry_points={
    }
)

