from setuptools import setup

with open("pimp/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open('requirements.txt') as fh:
    requirements = fh.read()
    requirements = requirements.split('\n')
    requirements = [requirement.strip() for requirement in requirements]
if 'git+http://github.com/automl/fanova.git@master' in requirements:
    idx = requirements.index('git+http://github.com/automl/fanova.git@master')
    requirements[idx] = 'fanova'

setup(
    name='PIMP',
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
    install_requires=requirements,
    url='',
    license='BSD 3-clause',
    author='biedenka',
    author_email='biedenka@cs.uni-freiburg.de',
    description='Package for automated Parameter Importance Analysis after Configuration.'
)
