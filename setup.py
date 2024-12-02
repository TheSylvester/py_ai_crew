from setuptools import setup, find_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py_ai_crew",
    version="0.1.0",
    author="Sylvester Wong",
    author_email="sylvester@thesylvester.ca",
    description="A hierarchical task execution framework for AI agents with goal-based validation and quality control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheSylvester/py_ai_crew",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.5.1",
        "pydantic-ai>=0.1.0",
        "asyncio>=3.4.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
            "ruff>=0.1.6",
        ],
    },
)
