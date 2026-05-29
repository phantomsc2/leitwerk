# Integration Guide

This page shows the typical pattern for wiring `leitwerk` into your own evaluation loop.

The intended call flow is sequential:

![Flowchart](flowchart.png)

---

## 1. Define a Parameter Schema

*~ Preflight Check ~*

```py
from dataclasses import dataclass

from leitwerk import parameter


@dataclass
class MyParams:
    attack_threshold: float = parameter()
    worker_limit: float = parameter(mean=66, scale=10, min=12)
```

`parameter(...)` defines the prior distribution for each value:

- `mean`: initial best guess
- `scale`: initial spread
- `min` and `max`: optional bounds

!!! tip
    Dictionary schemas are supported if you prefer string-based lookup:
    ```py
    from leitwerk import Parameter

    MyParams = {"a": Parameter(), "b": Parameter()}
    ```

!!! tip
    Use nested schemas to group parameters together.
    `leitwerk` understands tree structures.

!!! note
    Dataclass fields that are not parameters are passed through as constants.

## 2. Create the Optimizer

*~ Engine Ignition ~*

For automatic JSON persistence, use the `OptimizerSession` wrapper:

```py
from leitwerk import OptimizerSession

opt = OptimizerSession("params.json", MyParams)
```

If the session file already exists, it is loaded and reconciled automatically:

```pycon
>>> opt.restored
True
>>> opt.schema_diff
SchemaDiff(added=[], removed=[], changed=[], unchanged=['attack_threshold', 'worker_limit'])
```

!!! question
    [What happens when the schema changes?](faq.md#what-happens-when-the-schema-changes)

For an in-memory run, use `Optimizer` directly:

```py
from leitwerk import Optimizer

opt = Optimizer(MyParams)
```

If you are using other means of persistence, you can optionally restore it from state as well:

```py
schema_diff = opt.load(state)
```

Both `Optimizer` and `OptimizerSession` accept additional constructor arguments:

```py
from leitwerk import Optimizer

opt = Optimizer(MyParams, batch_size=10, seed=1234)
```

Available settings:

- `batch_size`: number of samples per batch / optimizer step
- `seed`: for reproducible runs

## 3. Sample a Candidate

*~ Liftoff ~*

```py
params = opt.ask()
```

```pycon
>>> params
MyParams(attack_threshold=-0.8312413125179872, worker_limit=59.407519238244)
```

For deterministic evaluation with `OptimizerSession`, use the optimized mean instead of sampling:

```pycon
>>> opt.mean()
MyParams(attack_threshold=0.0, worker_limit=66.0)
```

With `OptimizerSession`, optionally provide a context mapping for the current sample:

```py
context = {"map": "Goldenaura", "opponent": "Sharpy"}  # optional
params = opt.ask(context)
```

!!! question
    [What context should I provide?](faq.md#what-context-should-i-provide)

## 4. Report the Result

*~ Landing ~*

After evaluation, encode the outcome as one or more scalars:

```py
result = +1 if win else 0
report = opt.tell(result)
```

Binary win/loss alone tends to have little gradient to learn from.
Add smooth tie-breakers when possible:

```py
report = opt.tell((result, get_efficiency()))
```

Result handling:

- `opt.tell((a, b, c))` ranks results lexicographically with higher = better
- the first item is the main objective, later items are tie-breakers
- maximization is the default, flip the sign for loss objectives


!!! question
    [How should I choose the objective?](faq.md#how-should-i-choose-the-objective)

When using `OptimizerSession`, the JSON file is updated atomically on `tell`.
An `Optimizer` can be serialized with:

```py
state = opt.save()
```
