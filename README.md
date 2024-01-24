
  
  

#  SCOTCH v1.0.0 <img  src='images/icon.png'  align="right"  height="200" /></a>

  


## Cross-modal matching and integration of single-cell multi-omics data

  

### Penghui Yang<sup></sup>, Kaiyu Jin<sup></sup>, Lijun Jin<sup></sup>, ..., Xiaohui Fan*

  
  

SCOTCH is a computational method that leverages the optimal transport algorithm and a cell matching strategy to integrate scRNA-seq and scATAC-seq data. SCOTCH takes into account the adverse effects of **cell type abundance and cell number differences** on data integration during the calculation process, and **predicts cell pairing relationships** to meet the needs of downstream in-depth analysis.

  

![Image text](images/overview.png)

  

##  Installation of SCOTCH


[![pot 0.8.2](https://img.shields.io/badge/pot-0.8.2-blue)](https://pypi.org/project/POT/0.8.2/) [![numpy 1.22.4](https://img.shields.io/badge/numpy-1.22.4-green)](https://github.com/numpy/numpy/) [![pandas 1.4.3](https://img.shields.io/badge/pandas-1.4.3-yellowgreen)](https://github.com/pandas-dev/pandas/) [![scikit-learn 1.2.0](https://img.shields.io/badge/scikit--learn-1.2.0-yellow)](https://github.com/scikit-learn/scikit-learn/) [![scipy 1.8.1](https://img.shields.io/badge/scipy-1.8.1-orange)](https://github.com/scipy/scipy/) [![scanpy 1.9.1](https://img.shields.io/badge/scanpy-1.9.1-ff69b4)](https://pypi.org/project/scanpy/) [![anndata 0.7.5](https://img.shields.io/badge/anndata-0.7.5-purple)](https://github.com/scverse/anndata/) [![igraph 0.10.8](https://img.shields.io/badge/igraph-0.10.8-9cf)](https://github.com/igraph/igraph/) [![louvain 0.7.1](https://img.shields.io/badge/louvain-0.7.1-inactive)](https://pypi.org/project/louvain/0.7.1/) [![matplotlib 3.5.2](https://img.shields.io/badge/matplotlib-3.5.2-11adb1)](https://pypi.org/project/matplotlib/3.5.2/)



```

pip install scotch-sc

```

  

## Tutorials

  

We have applied SCOTCH on different tissues of multiple species, here we give step-by-step tutorials for application scenarios. And datasets in `.h5ad` fomat used can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1_LIBzeq-C028RC802ZwyfcqxjobEHTm1?usp=sharing).

  
  

* [Using SCOTCH to integrate SHARE-seq data of mouse cerebral cortex](tutorial/1.Chen-2019.ipynb)

  

* [Using SCOTCH to integrate human kidney snRNA-seq and snATAC-seq datasets](tutorial/2.Muto-2021.ipynb)

  

* [Using SCOTCH to integrate mouse cerebral cortex MERFISH and scATAC-seq datasets](tutorial/3.mouse_brain.ipynb)

  

## About

Should you have any questions, please feel free to contact the author of the manuscript, Mr. Penghui Yang (yangph@zju.edu.cn).

  

## References