<p align="center">
  <img src="https://raw.githubusercontent.com/phantomsc2/leitwerk/main/docs/logo.png" alt="Leitwerk">
</p>

<h1 align="center">Leitwerk</h1>
<p align="center">
  <em>Tune your magic numbers with stochastic optimization.</em>
</p>

`leitwerk` is a schema-based evolutionary optimizer for long-running training loops.

It offers:

- parameters as plain dataclasses in your code
- JSON checkpoints you can inspect and resume
- schema reconciliation, so you can keep developing and retain progress

## Links

- [Documentation](https://phantomsc2.github.io/leitwerk/)
- [Integration Guide](https://phantomsc2.github.io/leitwerk/guide/)
- [API Reference](https://phantomsc2.github.io/leitwerk/reference/api/)

## Installation

Requires: Python 3.11 - 3.14

```sh
pip install leitwerk
```

For a local development setup with testing, linting and `burnysc2`:

```sh
pip install -e .[dev,docs,benchmark]
```

The StarCraft II example also requires the game itself or Docker to be installed.

## Minimal Example

At base level, `leitwerk` is a sequential ask/tell function maximizer:

```py
from dataclasses import dataclass

from leitwerk import Optimizer, parameter

@dataclass
class Params:
    x: float = parameter()

opt = Optimizer(Params)
for _ in range(100):
    x = opt.ask().x
    fx = -(x - 1)**2
    opt.tell(fx)
```

```pycon
>>> opt.mean
Params(x=1.0000396079866551)
```

Everything else is wrapping for easier representation, bounds, typing and persistence.

## Example - StarCraft II Bot

For a real training loop showcasing most features, see [examples/train_sc2_bot.py](https://github.com/phantomsc2/leitwerk/blob/main/examples/train_sc2_bot.py):

- a simple worker rush bot with two parameters
- runs games continually against the built-in AI
- scores each game using win/loss and a combat heuristic
- persists state with `OptimizerSession` and plot results

Run it directly if you want to observe the games:

```sh
python examples/train_sc2_bot.py
```

Alternatively, use the headless Docker setup:

```sh
cd examples
docker compose up --build
```

On the first run, this will download and unpack the SC2 Linux installation inside the container.
Afterward, it will run significantly faster than the rendered game on the host.

### Training Results

The example saves progress in `examples/data` after each game:

- `params.json`: optimizer state
- `plot.png`: graphs showing parameter samples and results with a moving average
- `history.json`: helper file

<p align="center">
  <img src="https://raw.githubusercontent.com/phantomsc2/leitwerk/main/docs/example_plot.png" alt="Example Plot">
</p>

## Developer Commands

- `make fix`: auto-format
- `make check`: lint and test
- `make docs`: build docs
- `make docs-serve`: serve docs locally
