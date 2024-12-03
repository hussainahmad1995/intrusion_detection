# Network Intrusion

## Datasets

- Kitsune - <https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune>
- IDS2019 - <https://www.unb.ca/cic/datasets/ids-2018.html>
- UNSW - <https://research.unsw.edu.au/projects/unsw-nb15-dataset>

### Kitsune

#### Architecture

> The architecture of Kitsune is illustrated in the figure below:
>
> First, a feature extraction framework called AfterImage efficiently tracks the patterns of every network channel using damped incremental statisitcs, and extracts a feature vector for each packet. The vector captures the temporal context of the packet's channel and sender.
> Next, the features are mapped to the visible neurons of an ensemble of autoenoders (KitNET https://github.com/ymirsky/KitNET-py).
> Then, each autoencoder attempts to reconstruct the instance's features, and computes the reconstruction error in terms of root mean squared errors (RMSE).
> Finally, the RMSEs are forwarded to an output autoencoder, which acts as a non-linear voting mechanism for the ensemble.

## Methods
- SVM
- Transformers
