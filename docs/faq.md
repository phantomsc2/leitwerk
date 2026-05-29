# FAQ

## What happens when the schema changes?

`leitwerk` keeps learned state for parameters that still match and resets only what changed:

- parameters are identified by flattened names
- renaming a parameter resets that parameter
- changing `min` or `max` resets that parameter
- changing `mean` or `scale` changes the reset target, but does not force a reset
- adding or removing parameters does not reset the others

`Optimizer.load()` and `OptimizerSession.schema_diff` report what changed as a `SchemaDiff`.

## What context should I provide?

Context is for recurring conditions where good parameters shift.
It is a mapping: each key is a factor, and each distinct value inside that factor gets an individual offset model.
The active offsets are added to the global model, so common learning stays shared while context-specific effects can specialize.

Example:

```py
context = {"map": "Goldenaura", "opponent": "Sharpy"}
params = optimizer.ask(context)
```

Good factors are stable, low-cardinality values known before `ask()`.
For SC2 bots, useful factors include:

- map name
- opponent id
- opponent race, if known before `ask()`
- own race, for random bots

Avoid one-off values such as timestamps, match ids or raw scouting observations.

## How should I choose the objective?

For effective training, defining the objective matters more than the optimizer.

- put the primary objective first
- add tie-breakers for additional gradient information
- this is an encoding helper, not multi-objective / Pareto optimization
- only relative ranking matters, not absolute numeric values
- changing objectives mid-flight will leave the current batch with mixed signals
- split genuinely different goals into separate optimizers

## How does the optimizer work?

`leitwerk` provides a canonical xNES implementation.
Parameters are modeled as a multivariate normal distribution that is updated with natural-gradient steps.
The covariance matrix is estimated densely, initialization is diagonal.
Sampling uses internal variance reduction.

Bounds are modeled as unbounded latent normals with smooth bijective activations:

- one-sided (`min` or `max`): affine-transformed softplus
- two-sided (`min` and `max`): affine-transformed sigmoid

Reference Papers:

- [Exponential Natural Evolution Strategies](https://people.idsia.ch/~tom/publications/xnes.pdf)
