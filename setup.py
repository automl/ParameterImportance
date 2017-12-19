from setuptools import setup

with open("pimp/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setup(
    name='PIMP',
    version=version,
    packages=['pimp', 'pimp.epm', 'pimp.utils', 'pimp.utils.io', 'pimp.evaluator', 'pimp.importance',
              'pimp.configspace'],
    entry_points={
        'console_scripts': ['pimp=pimp.pimp:cmd_line_call'],
    },
    url='',
    license='BSD 3-clause',
    author='biedenka',
    author_email='biedenka@cs.uni-freiburg.de',
    description='Package for automated Parameter Importance Analysis after Configuration.'
)
