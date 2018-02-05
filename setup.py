from setuptools import setup, find_packages
from codecs import open
from os import path
from setuptools.command.install import install

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

class DownloadCorpora(install):
    def run(self):
        install.run(self)
        import spacy
        import nltk
        nltk.download('wordnet')
        spacy.cli.download('download', 'en')

class DownloadParaphraseModel(install):
    def run(self):
        install.run(self)
        from paraphaser.download_models import download_file_from_google_drive
        download_file_from_google_drive('19QDCd4UMgt3FtlYYwu0qZU3G1F9_XCvk', 
                                        'paraphrase-model.tar.gz')

setup(
    name='paraphraser',
    version='0.1.0',
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

    py_modules=['paraphraser.synonym_model', 'paraphraser.inference', 'paraphraser.download_models'],
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    #install_requires=['nltk', 'spacy', 'ipython'],
    install_requires=[],
    extras_require={

    },
    package_data={

    },
    data_files=[],
    entry_points={
    },
    cmdclass={
        'download_model': DownloadParaphraseModel
        #'download_corpora': DownloadCorpora
    }
)

