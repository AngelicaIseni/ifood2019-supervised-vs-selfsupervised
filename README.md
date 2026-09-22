![Food Recognition banner](docs/banner.svg)
# Food Recognition with Custom CNNs: Supervised vs Self-Supervised Pretraining

Fine-grained food classification on the [iFood-2019](https://www.kaggle.com/c/ifood-2019-fgvc6) dataset (251 classes, ~118K training images), comparing three custom convolutional networks trained from scratch and via transfer learning against two self-supervised pretraining strategies (colorization and SimCLR-style contrastive learning), all under a 10M-parameter budget.

Exam project for *Machine Learning for Modelling: Supervised Learning*, University of Milano-Bicocca, A.Y. 2025/26.

## Overview

We train and compare five encoders on the same food classification task:

- **Base net** — a plain 5-block CNN (1.04M params), trained from scratch
- **Medium net** — a residual CNN (5.03M params), trained from scratch, tested with dropout 0.3 and 0.5
- **Complex net** — EfficientNet-B0 pretrained on ImageNet (4.33M params), fine-tuned on the last blocks
- **Colorization encoder** — the medium net encoder, pretrained without labels to predict color from grayscale images
- **Contrastive encoder** — the medium net encoder, pretrained without labels with a SimCLR-style contrastive loss

The self-supervised encoders are frozen after pretraining and evaluated with linear and k-NN probes, to see how much class information a label-free representation captures on its own.

## Results

| Model | Accuracy | Precision | Recall | F1 | Weighted F1 |
|---|---|---|---|---|---|
| Base net | 53.1% | 52.2% | 51.3% | 50.6% | 52.6% |
| Medium net (dropout 0.3) | 57.8% | 58.7% | 56.4% | 56.1% | 57.7% |
| Medium net (dropout 0.5) | 58.7% | 59.3% | 57.8% | 57.2% | 58.6% |
| Complex net (EfficientNet-B0) | **64.9%** | 64.0% | 63.8% | 63.2% | 64.7% |
| Colorization (linear probe) | 27.6% | 26.3% | 26.5% | 25.8% | 27.1% |
| Contrastive (linear probe) | 31.6% | 29.8% | 30.3% | 29.5% | 30.9% |

Full metrics, training curves, confusion matrix and discussion of the results are in [`docs/Report.pdf`](docs/Report.pdf).

## Repository structure

```
.
├── notebooks/
│   ├── 1_food_utils.ipynb          # Shared utilities: data loading, Lightning wrapper, evaluation
│   ├── 2_DataExploration.ipynb     # Dataset download, class distribution, train/val split
│   ├── 3_BaseNet.ipynb             # Plain CNN, trained from scratch
│   ├── 4_MediumNet.ipynb           # Residual CNN, dropout 0.3
│   ├── 5_MediumNet2.ipynb          # Residual CNN, dropout 0.5
│   ├── 6_ComplexCNN.ipynb          # EfficientNet-B0, transfer learning
│   ├── 7_SSL_Colorization.ipynb    # Self-supervised pretraining: colorization pretext task
│   └── 8_SSL_Contrastive.ipynb     # Self-supervised pretraining: SimCLR-style contrastive task
├── docs/
│   ├── Report.pdf                  # Full written report
│   └── Food_Recognition.pptx       # Presentation slides
├── requirements.txt
└── README.md
```

## Getting started

All notebooks are designed to run on Google Colab.

1. Upload the `notebooks/` folder to your Google Drive (or add a shortcut to it).
2. Open each notebook and set the `PROJECT_DIR` variable to the path of that folder on your Drive.
3. Log in to your Kaggle account, go to **Profile → Your API Tokens → Generate New Token**, and download the token.
4. Paste your Kaggle API token where each notebook asks for `os.environ['KAGGLE_API_TOKEN']`.
5. Run `1_food_utils.ipynb` first — it defines the shared classes and functions the other notebooks import.
6. Run the remaining notebooks in order (`2` → `8`); each one downloads and extracts the dataset locally, since Colab's disk is wiped every session.

To run locally instead of on Colab, install the dependencies below and adapt the Drive/Colab-specific cells (dataset download, mounting) to your own environment.

## Requirements

See [`requirements.txt`](requirements.txt). Core stack: PyTorch + PyTorch Lightning for training, scikit-learn for probing/evaluation, scikit-image for the Lab color-space conversion used in the colorization task.

## Authors

- Angelica Iseni
- Kevin Del Gaudio

## References

- iFood 2019 Challenge at CVPR FGVC6 — [Kaggle competition](https://www.kaggle.com/c/ifood-2019-fgvc6)
- Kaur, P., Sikka, K., Wang, W., Belongie, S., Divakaran, A. *FoodX-251: a dataset for fine-grained food classification.* arXiv:1907.06167, 2019.
- Zhang, R., Isola, P., Efros, A. A. *Colorful Image Colorization.* ECCV, 2016.
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML, 2020.
- Falcon, W. and the PyTorch Lightning team. *PyTorch Lightning.* https://lightning.ai, 2019.
