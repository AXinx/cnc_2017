# Complex Network Classification with Convolutional Neural Network

This repository contains the official implementation of: ***[Complex network classification with convolutional neural network](https://ieeexplore.ieee.org/abstract/document/8954863)*** in Tsinghua Science and Technology, Volume 25, Number 4, 2020.

## Abstract
Classifying large-scale networks into several categories and distinguishing them according to their fine structures is of great importance to several real-life applications. However, most studies on complex networks focus on the properties of a single network and seldom on classification, clustering, and comparison between different networks, in which the network is treated as a whole. Conventional methods can hardly be applied on networks directly due to the non-Euclidean properties of data. In this paper, we propose a novel framework of Complex Network Classifier (CNC) by integrating network embedding and convolutional neural network to tackle the problem of network classification. By training the classifier on synthetic complex network data, we show CNC can not only classify networks with high accuracy and robustness but can also extract the features of the networks automatically. We also compare our CNC with baseline methods on benchmark datasets, which shows that our method performs well on large-scale networks.
<p align="center">
  <img src="./NEDMP_vis.png" width="450" title="hover text">
</p>

## Requirements
OS:
- Ubuntu

Python packages:
- troch==1.9.1
- torch-geometric>=2.0

## Code
We provide bash scripts (at `./bashs`) for generating all datasets used in our paper and running all experiments (as well as hyper parameters) conducted in our paper.

We implementate DMP using PyTorch at `./src/DMP/SIR.py`, and the code for NEDMP and baseline GNN at `src/model`.

## Citation

```
@article{xin2020complex,
  title={Complex network classification with convolutional neural network},
  author={Xin, Ruyue and Zhang, Jiang and Shao, Yitong},
  journal={Tsinghua Science and technology},
  volume={25},
  number={4},
  pages={447--457},
  year={2020},
  publisher={TUP}
}
```

