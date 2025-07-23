# ZipfFilter

A Python library for filtering sequences based on Zipf's Law. The Zipf Filter analyzes a sequence, fits a Zipf-Mandelbrot model to the frequency distribution, and removes or flags elements that deviate significantly from the expected pattern.

## Overview

Zipf's Law describes a power-law distribution often observed in natural datasets, such as word frequencies in text or event occurrences. The Zipf Filter leverages this principle to:
- Compute the frequency and rank of elements in a sequence.
- Fit a Zipf-Mandelbrot model to estimate the expected frequency distribution.
- Identify and filter outliers based on a customizable sensitivity threshold.
- Provide metrics about the filtering process.

This library is useful for applications like text data cleaning, anomaly detection in web analytics, security auditing, or any domain where Zipfian distributions are expected.

Read about the [Zipf Law](https://en.wikipedia.org/wiki/Zipf%27s_law) and the [Zipf-Mandelbrot law](https://en.wikipedia.org/wiki/Zipf%E2%80%93Mandelbrot_law)

## Installation

To install the Zipf Filter library, clone the repository and install the required dependencies:

```bash
git clone https://github.com/huolter/zipffilter
cd zipf-filter
pip install -r requirements.txt
```

### Requirements
- Python 3.6+
- Dependencies:
  - `numpy`
  - `scipy`

Install dependencies using the requiremnts.txt file or:
```bash
pip install numpy scipy
```

## Usage

The `ZipfFilter` class provides a simple interface for processing and filtering sequences.

### Example
```python
from zipf_filter import ZipfFilter

# Sample sequence (e.g., word frequencies)
sequence = ["apple", "banana", "apple", "cherry", "banana", "apple", "date", "date", "xyz", "xyz"]

# Initialize the filter with a sensitivity threshold
zf = ZipfFilter(sensitivity=0.2, use_relative_deviation=True)

# Filter the sequence
filtered_sequence, outliers = zf.filter_outliers(sequence)

# Print results
print("Filtered Sequence:", filtered_sequence)
print("Outliers:", outliers)
print("Metrics:", zf.get_metrics())
```

### Output
```
Filtered Sequence: ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'date', 'date']
Outliers: ['xyz']
Metrics: {'k': 3.1, 'a': 1.05, 'b': 2.8, 'sensitivity': 0.2, 'outliers': ['xyz'], 'num_outliers': 1}
```

### Parameters
- `sensitivity` (float, default: 0.2): Threshold for identifying outliers. Lower values are stricter.
- `use_relative_deviation` (bool, default: True): Use relative deviation instead of absolute deviation for outlier detection.

### Methods
- `process_sequence(sequence)`: Analyzes the sequence and fits the Zipf-Mandelbrot model.
- `filter_outliers(sequence)`: Filters the sequence and returns the filtered sequence and list of outliers.
- `get_metrics()`: Returns a dictionary with fitted parameters (`k`, `a`, `b`), sensitivity, and outlier information.

## Use Cases
- **Text Data Cleaning**: Remove unusual words or typos that deviate from expected frequency patterns.
- **Anomaly Detection**: Identify irregular patterns in web clicks, sales, or network traffic.
- **Data Quality Control**: Detect errors or anomalies in datasets where Zipf's Law applies.

## How It Works
1. **Frequency Analysis**: Counts occurrences of each element and ranks them by frequency.
2. **Zipf-Mandelbrot Fit**: Uses `scipy.optimize.curve_fit` to estimate parameters for the Zipf-Mandelbrot law: `frequency = k / (rank + b)^a`.
3. **Outlier Detection**: Calculates deviations from expected frequencies and flags elements exceeding the sensitivity threshold.
4. **Filtering**: Removes or flags outliers, preserving elements that conform to the Zipfian distribution.

## Limitations
- Small datasets may lead to unstable Zipf fits. The library includes fallback parameters to handle such cases.
- The Zipf-Mandelbrot model assumes a power-law distribution, which may not apply to all datasets.
- Sensitivity tuning may require experimentation for optimal results.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Contact
For questions or feedback, please open an issue or contact me at [x](https://x.com/walter_h_g_).