# nondecomposable-as-lp
This repository contains the reference code for our paper [Differentiable Optimization of Generalized
Nondecomposable Functions using Linear Programs](https://papers.nips.cc/paper/2021/file/f3f1b7fc5a8779a9e618e1f23a7b7860-Paper.pdf) (NeurIPS-2021)

## Requirements
* Python 3
* Pytorch 1.5+

## AUC optimization
For the four datasets (Cat&Dog, CIFAR10, CIFAR100, STL10) used in AUC optimization, we utilize the data and loaders provided by [Stochastic AUC Maximization with Deep Neural Networks](https://drive.google.com/drive/folders/1nPM6fmvN5fTsSaWsOcGFbhMVW7Fxso-Y). First download the four datasets from this link.

Then, use *run_auc_binary.sh* to run our algorithm for binary AUC optimization.

## Reference
If you find our work useful, please consider citing our paper.
```
@inproceedings{meng2021differentiable,
  title={Differentiable Optimization of Generalized Nondecomposable Functions using Linear Programs},
  author={Meng, Zihang and Mukherjee, Lopamudra and Wu, Yichao and Singh, Vikas and Ravi, Sathya N},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
