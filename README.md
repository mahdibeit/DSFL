# DSFL
Dynamic Sparsification for Federated Learning

# FLAC: Federated Learning with Autoencoder Compression and Convergence Guarantee

This is the implementation of the DSFL introduced in the paper submitted to ICCSPA2022. 

## Abstract
Federated Learning (FL) is considered the key, enabling approach for privacy-preserving, distributed machine learning (ML) systems. FL requires the periodic transmission of ML models from users to the server. Therefore, communication via resource-constrained networks is currently a fundamental bottleneck in FL, which is restricting the ML model complexity and user participation. One of the notable trends to reduce the communication cost of FL systems is gradient compression, in which techniques in the form of sparsification are utilized. However, these methods utilize a single compression rate for all users and do not consider communication heterogeneity in a realworld FL system. Therefore, these methods are bottlenecked by the worst communication capacity across users. Further, sparsification methods are non-adaptive and do not utilize the redundant, similar information across users’ ML models for compression. In this paper, we introduce a novel Dynamic Sparsification for Federated Learning (DSFL) approach that enables users to compress their local models based on their communication capacity at each iteration by using two novel sparsification methods: layer-wise similarity sparsification (LSS) and extended top-K sparsification. LSS enables DSFL to utilize the global redundant information in users’ models by using the Centralized Kernel Alignment (CKA) similarity for sparsification. The extended top-K model sparsification method empowers DSFL to accommodate the heterogeneous communication capacity of user devices by allowing different values of sparsification rate K for each user at each iteration. Our extensive experimental results 1 on three datasets show that DSFL has a faster convergence rate than fixed sparsification, and as the communication heterogeneity increases, this gap increases. Further, our thorough experimental investigations uncover the similarities of user models across the FL system.

## System Model
<p float="right">
  <img src="/images/DSFL.jpg" width="500" title="DSFL"/>
  
  <img src="/images/CKA.jpg" width="500" title="CKA" />

</p>

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Flower, Pytorch, and Tensorboard.

```bash
pip install Flower
pip install Pytorch
```

## Usage

```bash
python server.py

```
Then, run the client-***.sh file using you root directory and set the appropiate name instead of *** based on the dataset; e.g.:
```bash
run-mnistclients.sh

```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache](https://www.apache.org/legal/src-headers.html)
