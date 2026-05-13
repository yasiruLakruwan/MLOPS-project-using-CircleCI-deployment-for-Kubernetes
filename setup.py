import os
from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="CircleCI project",
    version="0.1",
    author="Yasiru lakruwan",
    install_requires=requirements,
    packages=find_packages()
)

