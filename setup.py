"""Setup configuration for AICtrlNet Community Edition."""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aictrlnet",
    version="1.0.0",
    author="AICtrlNet Team",
    author_email="team@aictrlnet.com",
    description="AICtrlNet Community Edition - Open source AI workflow orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bodaty/aictrlnet-community",
    license="MIT",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core web framework
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.25.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        # Database
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.29.0",
        "psycopg2-binary>=2.9.0",
        "alembic>=1.13.0",
        "greenlet>=3.0.0",
        # Cache
        "redis>=5.0.0",
        "aiocache>=0.12.0",
        # HTTP/Async
        "httpx>=0.25.0",
        "aiohttp>=3.9.0",
        "aiofiles>=23.0.0",
        "websockets>=12.0",
        # Auth/Security
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "bcrypt>=4.1.0",
        "cryptography>=41.0.0",
        "pyotp>=2.9.0",
        "qrcode[pil]>=7.4.0",
        "authlib>=1.3.0",
        # Utilities
        "python-multipart>=0.0.6",
        "email-validator>=2.1.0",
        "Jinja2>=3.1.0",
        "pyyaml>=6.0.0",
        "jsonschema>=4.20.0",
        "jsonpath-ng>=1.6.0",
        "json-repair>=0.25.0",
        "numpy>=1.24.0",
        "stripe>=7.0.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.22.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "opentelemetry-api>=1.22.0",
            "opentelemetry-sdk>=1.22.0",
            "opentelemetry-instrumentation-fastapi>=0.43b0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aictrlnet=main:main",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/Bodaty/aictrlnet-community/issues",
        "Documentation": "https://docs.aictrlnet.com",
        "Source": "https://github.com/Bodaty/aictrlnet-community",
    },
    keywords=[
        "ai",
        "workflow",
        "orchestration",
        "fastapi",
        "agents",
        "automation",
    ],
)
