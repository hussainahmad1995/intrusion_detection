# Network Intrusion

## Datasets

- Kitsune - <https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune> (Michael)
- IDS2019 - <https://www.unb.ca/cic/datasets/ids-2018.html> (Jiayi)
- UNSW - <https://research.unsw.edu.au/projects/unsw-nb15-dataset> (Hussain)

### Kitsune

#### Architecture

> The architecture of Kitsune is illustrated in the figure below:
>
> First, a feature extraction framework called AfterImage efficiently tracks the patterns of every network channel using damped incremental statisitcs, and extracts a feature vector for each packet. The vector captures the temporal context of the packet's channel and sender.
> Next, the features are mapped to the visible neurons of an ensemble of autoenoders (KitNET <https://github.com/ymirsky/KitNET-py>).
> Then, each autoencoder attempts to reconstruct the instance's features, and computes the reconstruction error in terms of root mean squared errors (RMSE).
> Finally, the RMSEs are forwarded to an output autoencoder, which acts as a non-linear voting mechanism for the ensemble.

## Methods

- SVM (Michael)
- Kernel Regression (Hussain)
- Transformers (Jiayi)

## Getting Started

All command should be run with `make` in the root of this repository.
To edit the latex paper, see [build-paper](#build-paper)

### Download

```sh
git clone https://github.com/hussainahmad1995/intrusion_detection
```

### Build Paper

Ensure you have a full TeX distribution and can invoke `latexmk`.
For building the paper, run `make pdf`, or just `make`.
Run `make pdf-clean` or just `make clean` to remove the intermediate files.

The PDF will be available in `bin/project-proposal.pdf`, all intermediate files
are stored in the `obj/` directory.

## HOWTOs

### How do I edit the paper latex source, build the paper, and view the pdf?

The paper source is all in one file main.tex. Add your changes to main.tex.
Then, build the pdf with `make pdf`.
Finally, open the pdf in your preferred pdf viewer (like Sioyek) in `bin/main.pdf`
