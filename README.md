[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7613150.svg)](https://doi.org/10.5281/zenodo.7613150)

# Unlearning Graph Classifiers with Limited Data Resources
A powerful and efficient method for graph unlearning when the size of training dataset is limited. Our accompany paper is as follows:

- [TheWebConf 2023] [Unlearning Nonlinear Graph Classifiers in the Limited Training Data Regime](https://arxiv.org/abs/2211.03216)

This work is developed based on our prior work certified graph unlearning. Feel free to check this [repository](https://github.com/thupchnsky/sgc_unlearn) and the following accompany papers for more details.

- [ICLR 2023] [Efficient Model Updates for Approximate Unlearning of Graph-Structured Data](https://openreview.net/forum?id=fhcu4FBLciL)
- [NeurIPS 2022 GLFrontier workshop] [Certified Graph Unlearning](https://openreview.net/forum?id=wCxlGc9ZCwi)

# Package Information
```
pytorch
scikit-learn
pytorch-dp
pytorch-geometric
tqdm
```

# Usage
### Project Setup
We assume the following project directory structure in our code:
```
<root>/
--> data/
--> results/
--> scripts/
```
If you have a different path for the datasets, you might need to change the utility function in `datasets.py`.

### Example

- To reproduce the results in Table 1 and 2 (test accuracy and running time of different backbone graph learning methods without unlearning):
```
./scripts/Exp1.sh
```
You can change the models or datasets that you want to test with by modifying `Exp1.sh`.

- To reproduce Figure 3 (test accuracy and running time of different unlearning methods, 10% sequential unlearning requests):
```
./scripts/Exp2.sh
```

- To reproduce Figure 4 (test accuracy and running time of different unlearning methods, 90% sequential unlearning requests):
```
./scripts/Exp3.sh
```

# Contact
Please contact Chao Pan (chaopan2@illinois.edu), Eli Chien (ichien3@illinois.edu) if you have any question.

# Citation
If you find our code or work useful, please consider citing our paper:
```
@inproceedings{
pan2023unlearning,
title={Unlearning Graph Classifiers with Limited Data Resources},
author={Chao Pan and Eli Chien and Olgica Milenkovic},
booktitle={The Web Conference},
year={2023}
}
```
