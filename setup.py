

from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES

# I have no freaking clue: https://stackoverflow.com/a/3042436
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(
    name                 = 'aspire',
    version              = '2024.9',
    description          = 'aspire',
    packages             = find_packages(exclude=['ez_setup', 'tests', 'tests.*']),
    package_data         = {'': ['pddlstream/*']},
    setup_requires       = ['setuptools_scm'], # https://stackoverflow.com/a/57932258
    include_package_data = True,
    install_requires     = [],
)
