# ZipfFilter

A Python library for filtering sequences based on Zipf's Law. ZipfFilter analyzes a sequence, fits a Zipf-Mandelbrot model to the observed frequency distribution, and removes elements that deviate significantly from the expected pattern.

## Overview

[Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law) describes a power-law distribution commonly observed in natural datasets — word frequencies in text, event counts in web analytics, and more. It states that an element's frequency is inversely proportional to its rank in the frequency table: the most common element appears roughly twice as often as the second most common, three times as often as the third, and so on.

ZipfFilter leverages the more general [Zipf-Mandelbrot law](https://en.wikipedia.org/wiki/Zipf%E2%80%93Mandelbrot_law) to:

- Compute the frequency and rank of every element in a sequence.
- Fit a Zipf-Mandelbrot model to estimate the expected frequency distribution.
- Identify and remove outliers based on a configurable sensitivity threshold.
- Expose fit metrics and outlier details for downstream inspection.

Typical applications include text data cleaning, anomaly detection in web analytics, security auditing, and any domain where a Zipfian distribution is a reasonable prior.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/huolter/zipffilter
cd zipffilter
pip install -r requirements.txt
```

**Requirements**

- Python 3.6+
- `numpy` ≥ 1.19.0
- `scipy` ≥ 1.5.0

## Usage

### Basic example

```python
from zipffilter import ZipfFilter

sequence = ["apple", "banana", "apple", "cherry", "banana", "apple", "date", "date", "xyz", "xyz"]

zf = ZipfFilter(sensitivity=0.2, use_relative_deviation=True)
filtered_sequence, outliers = zf.filter_outliers(sequence)

print("Filtered sequence:", filtered_sequence)
print("Outliers:", outliers)
print("Metrics:", zf.get_metrics())
```

**Output**

```
Filtered sequence: ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'date', 'date']
Outliers: ['xyz']
Metrics: {'k': 3.1, 'a': 1.05, 'b': 2.8, 'sensitivity': 0.2, 'outliers': ['xyz'], 'num_outliers': 1}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sensitivity` | `float` | `0.2` | Deviation threshold above which an element is considered an outlier. Lower values are stricter. |
| `use_relative_deviation` | `bool` | `True` | When `True`, deviation is normalized by the expected frequency, making the threshold scale-invariant. When `False`, raw absolute deviation is used. |

### Methods

| Method | Returns | Description |
|---|---|---|
| `process_sequence(sequence)` | `(freq_dict, sorted_frequencies)` | Counts element frequencies, assigns ranks in descending order, and fits the Zipf-Mandelbrot model. |
| `filter_outliers(sequence)` | `(filtered_sequence, outliers)` | Runs the full pipeline and returns the cleaned sequence alongside the list of removed elements. |
| `get_metrics()` | `dict` | Returns fitted model parameters (`k`, `a`, `b`), the configured sensitivity, the outlier list, and the outlier count. |

## How It Works

1. **Frequency analysis** — Element occurrences are counted with `collections.Counter` and sorted in descending order to assign stable ranks (rank 1 = most frequent).
2. **Model fitting** — `scipy.optimize.curve_fit` estimates the three parameters of the Zipf-Mandelbrot law: `frequency = k / (rank + b)^a`. If fitting fails (e.g., insufficient data), sensible defaults are used and a warning is emitted via the standard `logging` module.
3. **Outlier detection** — For each rank, the absolute difference between the observed and expected frequency is computed. With `use_relative_deviation=True` this value is normalized by the expected frequency. Elements whose deviation exceeds `sensitivity` are flagged as outliers.
4. **Filtering** — All occurrences of outlier elements are removed from the original sequence. Membership is tested against a `set` for O(1) lookup per element.

## Limitations

- **Small datasets** may produce unstable model fits. A fallback to default parameters is applied automatically, but results should be interpreted with caution.
- **Non-Zipfian data** — The Zipf-Mandelbrot model assumes a power-law distribution. Applying this filter to data that does not follow such a distribution will produce unreliable outlier classifications.
- **Sensitivity tuning** — The appropriate threshold depends on the dataset and use case. Some experimentation is typically required.

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

## Contact

For questions or feedback, open an issue or reach out on [X](https://x.com/walter_h_g_).
