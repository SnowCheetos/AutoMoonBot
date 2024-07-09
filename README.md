![](media/logo.png)

---------------

*This repository contains code and resources for automated algorithmic trading implementations. All content provided in this repository is for educational and informational purposes only. It is NOT intended to be a financial product or investment advice. Please carefully review the contents in* [`DISCLAIMERS`](DISCLAIMER.MD) *before proceeding further, by viewing, citing, using, or contributing to the code and resources provided in this repository, you acknowledge and agree to all terms, conditions, and disclaimers present and future, unless explicitly stated otherwise.*

# Introduction

## Intuitions
...

## Arguments Against Regression Methods

At first glence, it might be tempting to treat this as a *regression* problem and try to predict the price directly using non-linear regression methods such as *Kernel Methods*, *MLP*, or *autoregressive* methods such as *LSTM*. Although there are some merits in their uses, the price actions of any asset, albeit sequential and continuous in nature, can not *strictly*, or even *remotely* be dsecribed as a function of time, but rather a [*Stochastic Process*](https://en.wikipedia.org/wiki/Stochastic_process) (*ot more specifically, a* [*Markov Process*](https://en.wikipedia.org/wiki/Markov_chain)), where each time step, given its current or recent conditions, carries information about the potential future step in the form of a *probability distribution*. 

As an example, if d̶a̶d̶d̶y̶ Jerome Powell said "AI" four times during a news conference at 3PM, it becomes 1̶0̶0̶%̶ a bit more likely for *NVDA* to close higher that day, but it's not guaranteed. As an counter example, if Jimmy estimates that J̶i̶m̶m̶y̶'̶s̶ ̶0̶-̶D̶T̶E̶ ̶F̶D̶ ̶p̶u̶t̶ *NVDA* will close i̶n̶ ̶t̶h̶e̶ ̶m̶o̶n̶e̶y̶ at $\$150$ simply based on the fact that J̶i̶m̶m̶y̶ ̶h̶a̶s̶ ̶a̶ ̶s̶m̶o̶o̶t̶h̶ ̶b̶r̶a̶i̶n̶ it was $\$140$ at 3PM, we wouldn't consider Jimmy's argument to be very valid, which essentially is what many regression models tries to do, no offense to all the innocent r̶e̶g̶r̶e̶s̶s̶i̶o̶n̶ ̶m̶o̶d̶e̶l̶s̶ Jimmys in the world. The point being, at each time step $t$, rather than trying to predict the future price $p_{t+k}$, it is more ideal to try and predict the *transition probability distribution*. Or in other words, likely conditions of the near future, given observations of recent conditions. 

And hey, don't just take my word for it, [*Louis Bachelier*](https://en.wikipedia.org/wiki/Louis_Bachelier), credited to be the first person to model market price actions as stochastic processes (*and also the accidental discovery of Brownian Motion*), presented his now (*somewhat*) famous [*Bachelier Model*](https://en.wikipedia.org/wiki/Bachelier_model) on his PhD thesis (*Théorie de la spéculation, published 1900*) at the age of 30 s̶o̶ ̶m̶a̶y̶b̶e̶ ̶i̶t̶'̶s̶ ̶t̶i̶m̶e̶ ̶f̶o̶r̶ ̶y̶o̶u̶ ̶t̶o̶ ̶g̶e̶t̶ ̶a̶ ̶r̶e̶a̶l̶ ̶j̶o̶b̶, it was one of the first models that can be used to effectively determine the fair price of options, and it went on to inspire the creation of the infamous [*Black-Scholes Model*](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model), which for better or worse, was essential to this trillion dollar n̶a̶t̶i̶o̶n̶a̶l̶ ̶d̶e̶b̶t̶ financial derivatives industry we have today.

# Usage

*This project utilizes [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for Graph Neural Networks, I recommend visiting their site and install all dependencies according to their instructions.*

## Installation
...

## Quick Start
...

# Testing
Test scripts are available in the `tests` directory.

# Legacy

*This project has gone through many iterations ever since its creation in 2021, the current version is still super alpha. If you would like to view the legacy code and implementations for more insights, go to the* [`legacy`](legacy/README.MD) *directory, or use the quick links below.*

- [`Automated Trading with Policy Gradient`](legacy/policy_gradient/README.md)
- [`An Ensemble of Models Trading Together`](legacy/classical_ensemble/README.md)

# Contribute

Like software and finance? Have ideas on how to improve this project? Consider contributing! Visit [`CONTRIBUTE.MD`](docs/CONTRIBUTE.MD) for more information.


# Citation

If enjoyed the contents in this repository and found it helpful, please consider starring ⭐ s̶o̶ ̶I̶ ̶c̶a̶n̶ ̶f̶e̶e̶l̶ ̶b̶e̶t̶t̶e̶r̶ ̶a̶b̶o̶u̶t̶ ̶m̶y̶s̶e̶l̶f̶ to show your support. If you plan on using the contents in another project, please cite this repository. You can use the following `BibTeX` entry:

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