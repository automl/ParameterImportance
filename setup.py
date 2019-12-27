from setuptools import setup

with open("pimp/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setup(
    name='PyImp',
    version=version,
    packages=['pimp', 'pimp.epm', 'pimp.utils', 'pimp.utils.io', 'pimp.evaluator', 'pimp.importance',
              'pimp.configspace'],
    entry_points={
        'console_scripts': ['pimp=pimp.pimp:cmd_line_call'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    platforms=['Linux'],
    install_requires=[
        "numpy",
        "sklearn",
        "matplotlib",
        "ConfigSpace>=0.4",
        "scipy",
        "pyrfr>=0.8.0",
        "smac>=0.8.0",
        "fanova",
        "tqdm",
        "argcomplete",
        "pandas",
        "bokeh>=1.1.0",
    ],
    url='',
    license='BSD 3-clause',
    author='biedenka',
    author_email='biedenka@cs.uni-freiburg.de',
    description='Package for automated Parameter Importance Analysis after Configuration.'
)
