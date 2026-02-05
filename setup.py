from setuptools import setup, find_packages

setup(
    name="wbgapi360",
    version="0.2.6",
    description="Enterprise-grade World Bank Data Client for Humans & AI (MCP) Data360 API",
    long_description=open("README.md").read() if open("README.md") else "Official client for WBG Data360",
    long_description_content_type="text/markdown",
    author="Maykol Medrano",
    author_email="mmedrano2@uc.cl",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.23.0",
        "pandas>=1.3.0",
        "pydantic>=2.0.0",
        "nest_asyncio>=1.5.0",
        "fastmcp>=0.1.0",
    ],
    extras_require={
        "visual": ["matplotlib>=3.4.0", "seaborn>=0.11.0"],
        "map": ["geopandas>=0.9.0", "matplotlib>=3.4.0"],
        "dev": ["pytest", "pytest-asyncio"]
    },
    entry_points={
        "console_scripts": [
            "wbgapi360=wbgapi360.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
