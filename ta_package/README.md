# Temporal Abstraction Package

A Python package for temporal abstraction of time series data, providing various methods for symbolic representation of time series.

## Installation

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/yourusername/temporal_abstraction.git
```

## Usage

The package provides several methods for temporal abstraction:

```python
from temporal_abstraction import TemporalAbstraction
from temporal_abstraction.methods import equal_width, equal_frequency, sax, td4c

# Create a TemporalAbstraction instance
ta = TemporalAbstraction()

# Use different abstraction methods
# Equal Width
result = equal_width(data, n_bins=3)

# Equal Frequency
result = equal_frequency(data, n_bins=3)

# SAX (Symbolic Aggregate Approximation)
result = sax(data, n_bins=3)

# TD4C (Time Domain 4C)
result = td4c(data, n_bins=3)
```

## Methods Available

- Equal Width Binning
- Equal Frequency Binning
- SAX (Symbolic Aggregate Approximation)
- TD4C (Time Domain 4C)

## Requirements

- Python >= 3.6
- numpy >= 1.19.0
- pandas >= 1.0.0
- scikit-learn >= 0.24.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 