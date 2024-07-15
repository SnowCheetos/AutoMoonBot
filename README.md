![](media/logo_100.png)

---------------

This repository contains code and resources for automated algorithmic trading implementations. All content provided in this repository is for educational and informational purposes only. It is **NOT** intended to be a financial product or investment advice. By viewing, citing, using, or contributing to the contents and resources provided in this repository, you automatically acknowledge and agree to all terms and conditions found in [`DISCLAIMERS`](docs/DISCLAIMER.md), both present and future, unless explicitly stated otherwise.

# Table of Contents

Please note that this project is still super-pre-alpha, so hold your bags for now...

- [`Introduction`](#introduction)

- [`Usage`](#usage)

- [`Legacy`](#legacy)

- [`Contribution`](#contribution)

- [`Citation`](#citation)

# Introduction

See [here](automoonbot/README.MD)

## Intuitions

*Welp, you read faster than I can type, I feel the urge to type code instead of words, so... stay tuned*

# Usage

This project utilizes [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for Graph Neural Networks, I recommend visiting their site and install all dependencies according to their instructions.

## Installation

*Gimme a sec...*

## Quick Start

*Yeah no not yet...*

## Testing

Install `pytest` with `pip install -U pytest`.

Test scripts are available in the `tests` directory 

Run all tests with `pytest -v tests/`

To run tests for a specific module, run `pytest -v tests/{module}/...{submodules if necessary}/`

# Legacy

The current version of this project stands on the shoulders of... its past versions? Oh and giants too. Jokes aside, it went through a few significant iterations in the past, previous approaches used were very different. If you would like to view the legacy code and implementations for more insights, go to the `legacy` directory, or use the links below.

- [`Automated Trading with Policy Gradient`](legacy/policy_gradient/README.md)
- [`An Ensemble of Models Trading Together`](legacy/classical_ensemble/README.md)

# Contribution

Like software and finance? Have ideas on how to improve this project? Consider contributing! Refer to [`CONTRIBUTING`](docs/CONTRIBUTING.md) for more information.

# Citation

If enjoyed the contents in this repository, found it helpful, or at least entertaining, please consider starring ⭐ s̶o̶ ̶I̶ ̶c̶a̶n̶ ̶f̶e̶e̶l̶ ̶b̶e̶t̶t̶e̶r̶ ̶a̶b̶o̶u̶t̶ ̶m̶y̶s̶e̶l̶f̶ to show your support. If you plan on using the contents in another project, please cite this repository. You can use the following `BibTeX` entry:

```bibtex
@misc{SnowCheetos2024,
    author = {Zhihao (Cass) Sheng},
    title = {Algorithmic trading using heterogeneous graph neural network and reinforcement learning {AutoMoonBot}},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/SnowCheetos/AutoMoonBot}},
}
```