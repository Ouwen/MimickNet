from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas', 'scipy', 'polarTransform', 'numpy', 'tensorflow']

setup(
    name='MimickNet',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='''
    	Deep Learning Post-Processor for Creating Clinical Grade Ultrasound.'
    	@article{DBLP:journals/corr/abs-1908-05782,
		  author    = {Ouwen Huang and
		               Will Long and
		               Nick Bottenus and
		               Gregg E. Trahey and
		               Sina Farsiu and
		               Mark L. Palmeri},
		  title     = {MimickNet, Matching Clinical Post-Processing Under Realistic Black-Box
		               Constraints},
		  journal   = {CoRR},
		  volume    = {abs/1908.05782},
		  year      = {2019},
		  url       = {http://arxiv.org/abs/1908.05782},
		  archivePrefix = {arXiv},
		  eprint    = {1908.05782},
		  timestamp = {Mon, 19 Aug 2019 13:21:03 +0200},
		  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1908-05782},
		  bibsource = {dblp computer science bibliography, https://dblp.org}
		}
    ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    author='Ouwen Huang',
    author_email='ouwen.huang@duke.edu',    
)
