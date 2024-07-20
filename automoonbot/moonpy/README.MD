# AutoMoonBot

### Arguments Against Regression Methods

At first glence, it might be tempting to treat this as a *regression* problem and try to predict the price directly using non-linear methods such as Gaussian Regression, MLP, or autoregressive like LSTM or t̶h̶e̶ o̶v̶e̶r̶h̶y̶p̶e̶d̶ Transformers. 

Although there are some merits justifying their uses, the price actions of any asset, albeit sequential and continuous in nature, can not be dsecribed as simply a function of time. Rather, I would argue that it aligns much better to the description of a [Stochastic Process](https://en.wikipedia.org/wiki/Stochastic_process), or more specifically, a [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model). 

*A Markov chain or Markov process is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. Informally, this may be thought of as, "What happens next depends only on the state of affairs now"*.[[1]](https://en.wikipedia.org/wiki/Markov_chain#:~:text=A%20Markov%20chain%20or%20Markov%20process%20is%20a%20stochastic%20model%20describing%20a%20sequence%20of%20possible%20events%20in%20which%20the%20probability%20of%20each%20event%20depends%20only%20on%20the%20state%20attained%20in%20the%20previous%20event.%20Informally%2C%20this%20may%20be%20thought%20of%20as%2C%20%22What%20happens%20next%20depends%20only%20on%20the%20state%20of%20affairs%20now.%22) A hidden markov model is a markov process, where the underlying parameters describing the state transitions is not directly observable.

Thus, at each time step $t$, rather than trying to predict the future price $p_{t+k}$, it's more ideal to try and predict the *transition probability distribution*. Or in other words, likely conditions of the near future, given observations of the present (*in practice, recent*) conditions. 

And hey, don't just take my word for it, according to [wikipedia](https://en.wikipedia.org/wiki/Renaissance_Technologies#:~:text=In%201988%2C%20the%20firm%20established%20its%20most%20profitable%20portfolio%2C%20the%20Medallion%20Fund%2C%20which%20used%20an%20improved%20and%20expanded%20form%20of%20Leonard%20Baum%27s%20mathematical%20models%2C), the Medallion Fund's algorithm in 1988 used an improved and expanded form of [Baum–Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm), a type of EM algorithm for computing parameters from hidden markov models, it seems that the Medallion Fund also assumed, to some degree, that the market represents a hidden markov process.

As a side note, [Louis Bachelier](https://en.wikipedia.org/wiki/Louis_Bachelier), credited to be the first person to model market price actions as stochastic processes (*and also the accidental discovery of Brownian Motion*), presented his now (*somewhat*) famous [Bachelier Model](https://en.wikipedia.org/wiki/Bachelier_model) on his PhD thesis (*Théorie de la spéculation, published 1900*) at the age of 30 s̶o̶ ̶m̶a̶y̶b̶e̶ ̶i̶t̶'̶s̶ ̶t̶i̶m̶e̶ ̶f̶o̶r̶ ̶y̶o̶u̶ ̶t̶o̶ ̶g̶e̶t̶ ̶a̶ ̶r̶e̶a̶l̶ ̶j̶o̶b̶, it was one of the first models that can be used to effectively determine the fair prices of options, and it went on to inspire the creation of the infamous [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model), which for better or worse, was essential to this trillion dollar n̶a̶t̶i̶o̶n̶a̶l̶ ̶d̶e̶b̶t̶ financial derivatives industry we have today.

## Portfolio Modeling

Please refer to [here](session/README.md) for details.

## Data

...


## File Structure
```plaintext
backend/
    ├── data/
    │   ├── README.MD
    │   ├── __init__.py
    │   ├── element.py
    │   ├── nodes.py
    │   ├── edges.py
    │   ├── graph.py
    │   ├── feature.py
    │   ├── loader.py
    │   └── utils.py
    ├── session/
    │   ├── README.MD
    │   ├── __init__.py
    │   └── ...
    ├── server/
    │   ├── README.MD
    │   ├── __init__.py
    │   └── ...
    ├── __init__.py
    └── README.MD
```