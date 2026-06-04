# Temporal Abstraction Toolkit

A Python toolkit for converting numeric time-series data into **symbolic temporal representations**.

This repository implements several temporal abstraction methods that transform continuous time-series values into discrete states. These representations are useful for interpretable time-series analysis, sequential pattern mining, feature engineering, and symbolic machine-learning pipelines.

> Portfolio focus: **time-series representation learning, symbolic ML, temporal abstraction, and research tooling.**

## Motivation

Many time-series models operate directly on numeric values, but symbolic representations can make temporal behavior easier to mine, compare, and explain.

For example, instead of representing a signal only as raw values:

```text
[2.1, 2.4, 5.8, 6.2, 1.3]
```

we can abstract it into temporal states:

```text
low → low → high → high → low
```

This makes the data easier to use in downstream algorithms such as sequential pattern mining, temporal pattern discovery, and interpretable classification.

## Implemented Methods

- **Equal-width binning**  
  Divides the value range into equal intervals.

- **Equal-frequency binning**  
  Creates bins with approximately equal numbers of samples.

- **SAX — Symbolic Aggregate Approximation**  
  Converts a time series into a lower-dimensional symbolic representation.

- **TD4C — Time-Domain Class-Conscious Temporal Abstraction**  
  A supervised abstraction method that uses class labels to create more discriminative states.

## Repository Structure

```text
.
├── ta_package/              # package source code
├── FAGender.csv             # example dataset
├── test.ipynb               # example usage notebook
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yuval-haim/Hugobot2.git
cd Hugobot2
pip install -r requirements.txt
```

For development mode:

```bash
pip install -e .
```

## Usage

```python
import pandas as pd

from temporal_abstraction.methods import equal_width, equal_frequency, sax, td4c

data = pd.Series([2.1, 2.4, 5.8, 6.2, 1.3, 1.5, 7.0])

ew = equal_width(data, n_bins=3)
ef = equal_frequency(data, n_bins=3)
sx = sax(data, n_bins=3)
```

Example output:

```text
low, low, medium, high, low, low, high
```

## Example Pipeline

```text
Raw Time-Series
      ↓
Temporal Abstraction Method
      ↓
Symbolic State Sequence
      ↓
Pattern Mining / Classification / Visualization
```

## Suggested Results Section to Add

Add an example table after running the package on a dataset:

| Method | Supervised? | Output Type | Use Case |
|---|---:|---|---|
| Equal Width | No | Symbolic bins | simple baseline |
| Equal Frequency | No | Symbolic bins | balanced states |
| SAX | No | symbolic approximation | compact representation |
| TD4C | Yes | class-aware states | classification / discriminative abstraction |

## Why this is useful

Temporal abstraction is especially useful when the goal is not only prediction, but also **interpretability**. Instead of explaining a model through thousands of raw points, we can explain it using human-readable states and intervals.

## Future Work

- Add examples with multivariate time-series
- Add visualizations of cutpoints and state transitions
- Add benchmark notebooks comparing abstraction methods
- Add integration with sequential pattern mining algorithms
- Add tests and CI
