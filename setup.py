from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# long_description(後述)に、GitHub用のREADME.mdを指定
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tftime',
    packages=['tftime'],

    version='1.0.0',

    license='MIT',

    install_requires=['numpy', 'tensorflow'],

    author='komo135',
    author_email='komoootv@gmail.com',

    url='https://github.com/komo135/time-series-model', # パッケージに関連するサイトのURL(GitHubなど)

    description='This repository contains models that have been converted from image models and layers to time series.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='tftime',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
