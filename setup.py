from setuptools import setup

setup(
    author='Georg Ahnert',
    description='Create and evaluate pairwise comparison graphs',
    name='fairpair',
    version='0.1.0',
    license='MIT',
    packages=['fairpair'],
    install_requires=[
        'pandas>=1.3.5',
        'numpy>=1.20.3',
        'networkx>=2.6.3',
        'seaborn>=0.12.2',
        'scikit-learn>=1.0.2',
        'scipy>=1.3.2'
    ],
    python_requires='>=3.7'
)