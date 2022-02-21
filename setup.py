"""
Setup installation file for template.
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='template',
    version='0.1.0',
    description='template',
    long_description=long_description,
    url='https://github.com/jdmonaco/template',
    author='Joseph Monaco',
    author_email='jmonaco@hey.com',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10'
    ],
    keywords='python',
    packages=['template'])
