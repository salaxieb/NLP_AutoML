import pkg_resources
from setuptools import setup, find_packages

import pathlib

with open("Readme_pypi.md", "r") as fh:
    long_description = fh.read()

def requirements(filepath: str):
    with pathlib.Path(filepath).open() as requirements_txt:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]


setup(
    name='NLP AutoML',
    version='0.3',
    description='AutoML library for solving text -> label task',
    author="salaxieb",
    author_email='salaxieb.ildar@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=[
        'tests',
        'examples',
    ]),
    include_package_data=True,
    install_requires=requirements('requirements.txt'),
    extras_require={'dev': requirements('requirements.dev.txt')},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
