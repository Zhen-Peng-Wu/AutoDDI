## Requirements

```shell
Python == 3.7
PyTorch == 1.9.0
PyTorch Geometry == 2.0.3
rdkit == 2020.09.2
```

## Installation

```shell
conda create -n AutoDDI python=3.7
conda activate AutoDDI
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric==2.0.3
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
conda install -c rdkit rdkit
```

## Dataset Preparation

```shell
cd AutoDDI-master/
```

download the datasets from Google Drive:
```shell
https://drive.google.com/open?id=1h_lYXoPEuLygOsMD9yyLmVqkeEGfob5r
```

then unzip the file
```shell
unzip AutoDDI_dataset.zip
```

## Quick Start

A quick start example is given by:

```shell
$ python autoddi_main.py
```

An example of auto search is as follows:

first, modify the code in ```set_config.py``` file
```shell
gnn_parameter['mode'] = 'test'
```
as
```shell
gnn_parameter['mode'] = 'search'
```

then, run the ```autoddi_main.py```
```shell
$ python autoddi_main.py
```
