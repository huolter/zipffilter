import numpy as np
from scipy.optimize import curve_fit
from collections import Counter
import logging

class ZipfFilter:
    def __init__(self, sensitivity=0.2, use_relative_deviation=True):
        """
        Initialize the Zipf Filter.
        
        Parameters:
        - sensitivity (float): Threshold for filtering outliers (default: 0.2).
        - use_relative_deviation (bool): Use relative deviation instead of absolute (default: True).
        """
        self.sensitivity = sensitivity
        self.use_relative_deviation = use_relative_deviation
        self.params = None
        self.frequencies = None
        self.ranks = None
        self.filtered_sequence = None
        self.outliers = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def zipf_mandelbrot(self, rank, k, a, b):
        """
        Zipf-Mandelbrot law: frequency = k / (rank + b)^a
        """
        return k / (rank + b) ** a

    def fit_zipf(self, frequencies, ranks):
        """
        Fit the Zipf-Mandelbrot model to the frequency data.
        """
        try:
            # Initial guess for parameters [k, a, b]
            initial_guess = [max(frequencies), 1.0, 2.7]
            params, _ = curve_fit(self.zipf_mandelbrot, ranks, frequencies, p0=initial_guess)
            return params
        except RuntimeError as e:
            self.logger.warning(f"Curve fitting failed: {e}. Using default parameters.")
            return [max(frequencies), 1.0, 2.7]

    def process_sequence(self, sequence):
        """
        Process the input sequence and fit the Zipf-Mandelbrot model.
        
        Parameters:
        - sequence: List or iterable of elements (e.g., words, counts).
        
        Returns:
        - frequencies: Array of observed frequencies.
        - ranks: Array of ranks for each unique element.
        """
        if not sequence:
            raise ValueError("Input sequence is empty.")
        
        # Count frequencies of elements
        freq_dict = Counter(sequence)
        self.frequencies = np.array(list(freq_dict.values()))
        sorted_freq = sorted(self.frequencies, reverse=True)
        self.ranks = np.arange(1, len(sorted_freq) + 1)
        
        # Fit Zipf-Mandelbrot model
        self.params = self.fit_zipf(sorted_freq, self.ranks)
        
        return freq_dict, sorted_freq

    def filter_outliers(self, sequence):
        """
        Filter the sequence by removing elements that deviate from the Zipf-Mandelbrot model.
        
        Parameters:
        - sequence: List or iterable of elements.
        
        Returns:
        - filtered_sequence: Sequence with outliers removed.
        - outliers: List of elements identified as outliers.
        """
        if not sequence:
            raise ValueError("Input sequence is empty.")
        
        freq_dict, sorted_freq = self.process_sequence(sequence)
        
        # Calculate expected frequencies using the fitted model
        expected_freq = self.zipf_mandelbrot(self.ranks, *self.params)
        
        # Calculate deviations
        deviations = np.abs(sorted_freq - expected_freq)
        if self.use_relative_deviation:
            deviations = deviations / expected_freq
        
        # Determine outliers
        threshold = self.sensitivity
        outlier_indices = np.where(deviations > threshold)[0]
        
        # Map ranks back to elements
        rank_to_element = {i + 1: elem for i, (elem, _) in enumerate(
            sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))}
        
        # Identify outlier elements
        self.outliers = [rank_to_element[rank] for rank in self.ranks[outlier_indices]]
        
        # Filter sequence
        self.filtered_sequence = [item for item in sequence if item not in self.outliers]
        
        return self.filtered_sequence, self.outliers

    def get_metrics(self):
        """
        Return metrics about the Zipf fit and filtering process.
        
        Returns:
        - dict: Contains fitted parameters, threshold, and outliers.
        """
        if self.params is None:
            raise ValueError("No data processed yet. Run filter_outliers first.")
        
        return {
            "k": self.params[0],
            "a": self.params[1],
            "b": self.params[2],
            "sensitivity": self.sensitivity,
            "outliers": self.outliers,
            "num_outliers": len(self.outliers) if self.outliers else 0
        }