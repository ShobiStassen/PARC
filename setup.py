import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='parc',
    version='0.34',  #May 28,2024
    packages=['parc',],
    license='MIT',
    author_email='shobana.venkat88@gmail.com',
    url='https://github.com/ShobiStassen/PARC',
    setup_requires=['numpy', 'pybind11'],
    install_requires=[
        'pybind11', 'numpy', 'scipy', 'pandas', 'hnswlib', 'igraph',
        'leidenalg>=0.7.0', 'umap-learn'
    ],
    extras_require={
        "dev": ["pytest", "scikit-learn"]
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
