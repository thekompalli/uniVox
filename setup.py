"""
PS-06 Competition System Setup
Installation script for the PS-06 Language Agnostic Speaker Identification & Diarization System
"""
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python 3.9+
if sys.version_info < (3, 9):
    raise RuntimeError("PS-06 System requires Python 3.9 or higher")

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]
    return []

# Core requirements
install_requires = read_requirements('requirements.txt')

# Development requirements
dev_requires = [
    'pytest>=7.4.3',
    'pytest-asyncio>=0.21.1',
    'pytest-cov>=4.1.0',
    'pytest-mock>=3.12.0',
    'pytest-xdist>=3.5.0',
    'pytest-benchmark>=4.0.0',
    'black>=23.11.0',
    'isort>=5.12.0',
    'flake8>=6.1.0',
    'mypy>=1.7.1',
    'pre-commit>=3.6.0',
    'coverage>=7.3.2',
]

# Documentation requirements
docs_requires = [
    'sphinx>=7.2.6',
    'sphinx-rtd-theme>=1.3.0',
    'mkdocs>=1.5.3',
    'mkdocs-material>=9.4.8',
    'myst-parser>=2.0.0',
]

# Performance and profiling
perf_requires = [
    'line-profiler>=4.1.1',
    'memory-profiler>=0.61.0',
    'py-spy>=0.3.14',
    'nvidia-ml-py3>=7.352.0',
]

# Cloud deployment
cloud_requires = [
    'boto3>=1.34.0',
    'google-cloud-storage>=2.10.0',
    'azure-storage-blob>=12.19.0',
]

# Competition specific
competition_requires = [
    'python-dotenv>=1.0.0',
    'pyyaml>=6.0.1',
    'click>=8.1.7',
    'typer>=0.9.0',
    'rich>=13.7.0',
]

# Get version
def get_version():
    """Get version from __init__.py"""
    version_file = this_directory / "src" / "__init__.py"
    if version_file.exists():
        for line in version_file.read_text().splitlines():
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "1.0.0"

setup(
    name="ps06-competition-system",
    version=get_version(),
    author="PS-06 Competition Team",
    author_email="team@ps06-system.com",
    description="Language Agnostic Speaker Identification & Diarization System for PS-06 Competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ps06-system",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/ps06-system/issues",
        "Documentation": "https://ps06-system.readthedocs.io/",
        "Source Code": "https://github.com/your-org/ps06-system",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "configs": ["**/*.yaml", "**/*.yml", "**/*.json"],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
        'docs': docs_requires,
        'perf': perf_requires,
        'cloud': cloud_requires,
        'competition': competition_requires,
        'all': dev_requires + docs_requires + perf_requires + cloud_requires + competition_requires,
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'ps06-server=src.api.main:main',
            'ps06-worker=src.tasks.worker:main',
            'ps06-process=src.cli.process:main',
            'ps06-setup=src.cli.setup:main',
            'ps06-validate=src.cli.validate:main',
            'ps06-benchmark=src.cli.benchmark:main',
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    
    # Keywords
    keywords=[
        "speech processing", "speaker diarization", "speaker identification",
        "automatic speech recognition", "machine translation", "multilingual",
        "competition", "ps06", "indian languages", "audio processing"
    ],
    
    # Minimum requirements check
    zip_safe=False,
    
    # Platform compatibility
    platforms=["linux", "darwin", "win32"],
    
    # License
    license="MIT",
    
    # Additional metadata
    maintainer="PS-06 System Maintainers",
    maintainer_email="maintainers@ps06-system.com",
    
    # Custom commands
    cmdclass={},
)

# Post-installation checks and setup
def post_install():
    """Post-installation setup"""
    print("\n" + "="*60)
    print("PS-06 Competition System Installation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure your settings")
    print("2. Download models: ./scripts/download_models.sh")
    print("3. Setup environment: ./scripts/setup_environment.sh")
    print("4. Run tests: ./scripts/run_tests.sh")
    print("5. Start the system: ps06-server")
    print("\nDocumentation: https://ps06-system.readthedocs.io/")
    print("Support: https://github.com/your-org/ps06-system/issues")
    print("\nFor competition submission help:")
    print("  ps06-validate --help")
    print("  ps06-process --help")
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run setup
    setup()
    
    # Run post-installation
    if "install" in sys.argv:
        post_install()