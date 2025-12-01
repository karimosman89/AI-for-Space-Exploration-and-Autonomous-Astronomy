"""
Setup configuration for AI Space Exploration package.
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='space-ai-explorer',
    version='1.0.0',
    author='Karim Osman',
    author_email='karim.osman@example.com',
    description='AI solutions for space exploration and autonomous astronomy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.5.0',
        'scikit-learn>=1.0.0',
        'astropy>=5.0.0',
        'fastapi>=0.100.0',
        'streamlit>=1.25.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'requests>=2.31.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'ml': [
            'tensorflow>=2.12.0',
            'transformers>=4.30.0',
            'stable-baselines3>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'space-ai=src.cli:main',
            'space-ai-train=src.cli:train',
            'space-ai-serve=src.api.main:serve',
        ],
    },
    include_package_data=True,
    keywords='ai machine-learning space-exploration astronomy satellite-imagery deep-learning',
    project_urls={
        'Bug Reports': 'https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy/issues',
        'Source': 'https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy',
        'Documentation': 'https://github.com/karimosman89/AI-for-Space-Exploration-and-Autonomous-Astronomy/tree/main/docs',
    },
)
