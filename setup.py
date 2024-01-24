from setuptools import setup, find_packages

setup(
    name='scotch',
    version='1.0.0',
    author='Penghui Yang',
    author_email='yangph@zju.edu.cn',
    description='SCOTCH: Single-Cell multi-modal integration using Optimal Transport and Cell matCHing',
    long_description='SCOTCH is a Python package for the integration of single-cell gene expression and chromatin accessibility data, leveraging the Optimal Transport algorithm and a cell matching strategy. It provides a matching-based unsupervised integration method for exploring cellular regulatory mechanisms.',
    long_description_content_type='text/markdown',
    url='https://github.com/ZJUFanLab/SCOTCH',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pot==0.8.2',
        'numpy==1.22.4',
        'pandas==1.4.3',
        'scikit-learn==1.2.0',
        'scipy==1.8.1',
        'scanpy==1.9.1',
        'anndata==0.7.5',
        'igraph==0.10.8',
        'louvain==0.7.1',
        'matplotlib==3.5.2',
        'scikit-misc==0.1.4',
        'leidenalg==0.10.0',
        'pamona==0.1.0',
        'unioncom==0.4.0'
        
    ],
)