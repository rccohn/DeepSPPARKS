from setuptools import find_packages, setup

setup(
        name='candidate_grains_spparks', # still imports with import spparks2graph,
        # TODO figure out better name 
        packages=find_packages(),
        version='0.0.1',
        description='Graph datastructures for data generated with SPPARKS-meso candidate grain abnormal grain growth simulations',
        author='Ryan Cohn',
        license='MIT',
        python_requires='>=3.6'
        )
