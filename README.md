![](media/logo_100.png)

---------------

This repository contains code and resources for automated algorithmic trading implementations. All content provided in this repository is for educational and informational purposes only. It is **NOT** intended to be a financial product or investment advice. By viewing, citing, using, or contributing to the contents and resources provided in this repository, you automatically acknowledge and agree to all terms and conditions found in [`DISCLAIMERS`](docs/DISCLAIMER.md), both present and future, unless explicitly stated otherwise.

# Table of Contents

**Please note that this project is still super-pre-alpha, so hold your bags for now...**

- [`Introduction`](#introduction)

- [`Usage`](#usage)

- [`Legacy`](#legacy)

- [`Contribution`](#contribution)

- [`Citation`](#citation)

# Introduction

See more [here](automoonbot/README.MD)

## Intuitions

[The Medallion Fund](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html) is one of the core inspirations for this project. It showed that quantitative trading, when executed perfectly, has enormous potential for success and profit. The fund averaged an astonishing annualized return of **63.3%** for its duration, and it didn't have a single negative year, even during the **2000 dot-com bubble** and **2008 financial crisis**. 

## Approach

The Medallion Fund publically stated numerous times that they collect nearly all publically available data, anything that could remotely affect the price actions of an asset was included. This may not be as difficult as it sounds, since the market is inheriently influenced by the masses of investors, most of whom have access to similar amount of information. Therefore, it can be assumed that any data which can be gathered publically would have been reviewed by some investors, which inheriently means it carries some degree of influence over the price actions. However, simply having the data does not mean it can be used to make consistent profits, otherwise anyone who knows a bit of software can build a scraper and would be filthy rich. Rather, the difficult part is to make effective use of the data, which can come in a variety of forms, making it difficult to construct a single comprehensive representation. For instance, imagine one obtains news articles in the form of raw `HTML` contents, and federal interest rates as time series, it would be difficult to design a model that could take in all of them and extract meaningful information.

## Graphs

...

## Reinforcement Learning

...

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