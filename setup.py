"""
Setup script for Cervical Cytology AI System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cervical-cytology-ai",
    version="1.0.0",
    author="Cervical AI Team",
    author_email="contact@example.com",
    description="AI-based cervical cell diagnosis system using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xsxweri/cervical-ai-diagnosis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cervical-train=scripts.train:main",
            "cervical-evaluate=scripts.evaluate:main",
            "cervical-predict=scripts.predict:main",
        ],
    },
)
