![](media/logo_100.png)

-----

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

## Inspiration

[The Medallion Fund](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html) is one of the core inspirations for this project. It showed that quantitative trading, when executed perfectly, has enormous potential for success and profit. The fund averaged an astonishing annualized return of **63.3%** for its duration, and it didn't have a single negative year, even during the **2000 dot-com bubble** and **2008 financial crisis**. This level of success caused many to question the highly controversial [Efficient Market Hypothesis](https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp), whose definition will squash the hopes and dreams of any u̶n̶e̶m̶p̶l̶o̶y̶e̶d̶ ̶e̶c̶o̶n̶ ̶m̶a̶j̶o̶r̶ traders who aims to beat the market with technical analysis.

Fortunately, t̶h̶e̶i̶r̶ ̶p̶a̶r̶e̶n̶t̶s̶ ̶s̶t̶i̶l̶l̶ ̶l̶o̶v̶e̶ ̶t̶h̶e̶m̶ the efficient market hypothesis has been proven wrong on many occasions (*although I question that every time I make a trade*). Still, a majority of the funds do not manage to beat the market consistently, not to mention retail investors. Point being, it's possible to beat the market, but possible doesn't remotely equate probable in this case. Now, I'm no [Jim Simons](https://en.wikipedia.org/wiki/Jim_Simons) d̶u̶h̶ (*peace be with him* 🙏🏼), and the Medallion Fund keeps most details of its methodologies confidential, but Jim Simons and others have provided some valuable information to the public, maybe just enough to make m̶e̶ ̶r̶i̶c̶h̶ a difference. I have summarized them into four points (*note that a lot of them are my personal interpretations*)

### 1. It's Just Math
-----
Algorithmic trading is at the core of the Medallion Fund, with Jim Simons himself being a renowned mathematician in his earlier life. Their approach was almost purely from the numerical perspective, rather than finance. S̶o̶r̶r̶y̶ ̶p̶a̶l̶,̶ ̶t̶h̶a̶t̶ ̶e̶c̶o̶n̶ ̶d̶e̶g̶r̶e̶e̶ ̶t̶h̶a̶t̶ ̶y̶o̶u̶ ̶"̶w̶o̶r̶k̶e̶d̶ ̶h̶a̶r̶d̶"̶ ̶f̶o̶r̶ ̶i̶s̶ ̶n̶o̶t̶ ̶g̶o̶n̶n̶a̶ ̶p̶a̶y̶ ̶o̶f̶f̶ ̶t̶h̶e̶ ̶w̶a̶y̶ ̶y̶o̶u̶ ̶t̶h̶i̶n̶k̶

### 2. Everything, Anything
-----
Jim Simons have stated explicitly in public that the fund makes extensive use of publically available data to train their model. Anything from news to interest rates were taken into consideration, and much of it **before** the wide adaption of the internet, so they spent a lot of time copying data by hand. S̶o̶ ̶i̶t̶'̶s̶ ̶t̶i̶m̶e̶ ̶f̶o̶r̶ ̶y̶o̶u̶ ̶t̶o̶ ̶s̶t̶o̶p̶ ̶w̶h̶i̶n̶i̶n̶g̶ ̶a̶b̶o̶u̶t̶ ̶b̶e̶i̶n̶g̶ ̶b̶r̶o̶k̶e̶

### 3. Win Small or Lose Big
-----
For the duration of the fund's existence, their model had an average win rate of $`50.75\%`$. That's right, they operated on a tiny $`0.75\%`$ edge over the market, which was enough to turn $`\$100`$ in $`\text{1982}`$ to $`\$88,190,084,543`$ in $`\text{2024}`$. Instead of trying to count cards at Blackjack, they just opened a casino. Staying consistent over millions of trades turned out to be much more advantageous than trying to win big on a few trades.

### 4. Focus on The Right Thing
-----
Many people are willing to take on exceptional risks in pursue of, at least theoretically, higher possible returns, l̶i̶k̶e̶ ̶t̶h̶a̶t̶ ̶o̶n̶e̶ ̶w̶e̶e̶k̶l̶y̶ ̶F̶D̶ ̶y̶o̶u̶ ̶l̶o̶s̶t̶ ̶y̶o̶u̶r̶ ̶l̶u̶n̶c̶h̶ ̶m̶o̶n̶e̶y̶ ̶o̶n̶ ̶l̶a̶s̶t̶ ̶F̶r̶i̶d̶a̶y̶, and in some cases, risk taking is even being glorified (*not pointing any fingers* 👉🏻 [`👀`](https://www.reddit.com/r/wallstreetbets/)). Too often do m̶y̶s̶e̶l̶f̶ traders focus solely on potential (*not even real*) profits while completely ignoring the associated risk. The Medallion Fund, on the other hand, placed a tremendous emphasis on risk reduction. The fund consistently hedged the portfolio near perfectly, which enabled them to use margins that would be otherwise considered risky. 

## Approach

The Medallion Fund publically stated numerous times that they collect nearly all publically available data, anything that could remotely affect the price actions of an asset was included. This may not be as difficult as it sounds, since the market is inheriently influenced by the masses of investors, most of whom have access to similar amount of information. Therefore, it can be assumed that any data which can be gathered publically would have been reviewed by some investors, which inheriently means it carries some degree of influence over the price actions. However, simply having the data does not mean it can be used to make consistent profits, otherwise anyone who knows a bit of software can build a scraper and would be filthy rich. Rather, the difficult part is to make effective use of the data, which can come in a variety of forms, making it difficult to construct a single comprehensive representation. For instance, imagine one obtains news articles in the form of raw `HTML` contents, and federal interest rates as time series, it would be difficult to design a model that could take in all of them and extract meaningful information.

See [**here**](automoonbot/README.MD#arguments-against-regression-methods) to see arguments against using simple regression models. Readme will be better organized in the near future.

## Graphs

*Gimme a sec...*

## Reinforcement Learning

*...*

# Usage

This project utilizes [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for Graph Neural Networks, I recommend visiting their site and install all dependencies according to their instructions.

## Installation

*Gimme a min...*

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