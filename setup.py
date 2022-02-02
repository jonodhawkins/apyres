from setuptools import setup, find_packages

setup(
    name="pyapres",
    packages=find_packages(),
    package_dir={"":"src"},
    include_package_data=True,
    version='0.0.1',
    description='Python library for processing of ApRES radar data',
    author='Jono Hawkins',
    author_email='jonathan.hawkins.17@ucl.ac.uk'
)
