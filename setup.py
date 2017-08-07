from distutils.core import setup

setup(
    name='PIMP',
    version='v0.0.1',
    packages=['pimp', 'pimp.epm', 'pimp.utils', 'pimp.utils.io', 'pimp.evaluator', 'pimp.importance',
              'pimp.configspace'],
    url='',
    license='BSD 3-clause',
    author='biedenka',
    author_email='biedenka@cs.uni-freiburg.de',
    description='Package for automated Parameter Importance Analysis after Configuration.'
)
