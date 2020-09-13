from setuptools import setup, find_packages
import os, sys

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="nlu_flow",
    version="0.0.2",
    description="Natural Language Flow",
    author="Cheesama",
    install_requires=[required],
    packages=find_packages(exclude=["docs", "tests", "tmp", "data"]),
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
)

os.system('pip install -r requirements_model.txt')
