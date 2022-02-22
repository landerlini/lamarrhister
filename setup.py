from distutils.core import setup
import setuptools
from glob import glob

with open ("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name='lamarrhister',
    version='0.1',
    description='Helper package to fill histograms in Dirac jobs',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lucio Anderlini',
    author_email='Lucio.Anderlini@fi.infn.it',
    url='https://github.com/landerlini/lamarrhist',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programmin Language :: Python :: 3"
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "uproot>=4.0"
    ],
    extras_require={
        'reports': [
            'html-reports>=0.2',
            'matplotlib>=3.2',
            'scipy>=1.0',
        ]
    },
    entry_points={  # Optional
        'console_scripts': [
            'fill-hist=lamarrhister.__main__:main',
            'merge-hist=lamarrhister.merge_hists:merge_hists',
            'make-report=lamarrhister.make_report:make_report',
        ],
    },
)
