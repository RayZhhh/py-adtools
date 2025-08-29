from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='py-adtools',
    version='0.1.7',
    author='Rui Zhang',
    author_email='rzhang.cs@gmail.com',
    description='Useful tools for parsing and evaluating Python programs for LLM-based algorithm design.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RayZhhh/py-adtools',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.10',
    install_requires=['psutil', 'openai'],
)
