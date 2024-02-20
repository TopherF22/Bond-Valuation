import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class YieldCurve:
    def __init__(self, maturities, rates):
        """
        Initialize the YieldCurve class with maturities and their corresponding rates.
        """
        self.maturities = maturities
        self.rates = rates
        self._spline = None

    @property
    def maturities(self):
        return self._maturities

    @maturities.setter
    def maturities(self, value):
        if not all(m > 0 for m in value):
            raise ValueError("All maturities must be positive.")
        self._maturities = np.array(value)
        self._invalidate_spline()

    @property
    def rates(self):
        return self._rates

    @rates.setter
    def rates(self, value):
        if not all(r >= 0 for r in value):
            raise ValueError("Rates must be non-negative.")
        self._rates = np.array(value)
        self._invalidate_spline()

    def _invalidate_spline(self):
        self._spline = None

    def _fit_curve(self):
        """Fit a cubic spline to the provided maturities and rates."""
        if len(self._maturities) < 4 or len(self._rates) < 4:
            raise ValueError("At least four data points are required for cubic spline interpolation.")
        if self._spline is None:
            self._spline = CubicSpline(self._maturities, self._rates)

    # this is currently unused, could be useful for future development
    @staticmethod
    def format_maturity_labels(maturities):
        """
        Format the maturity values to human-readable labels.
        """
        labels = []
        for m in maturities:
            if m < 1:
                # Convert to months
                month = int(round(m * 12))
                labels.append(f'{month}M')
            elif m == 1:
                labels.append('1Y')
            else:
                # Convert to years and add 'Y' suffix
                year = int(round(m))
                labels.append(f'{year}Y')
        return labels

    def get_rate(self, maturity):
        """
        Calculate the interpolated rate for a given maturity.
        """
        if self._spline is None:
            self._fit_curve()
        return self._spline(maturity)
    
    def plot_curve(self, start=0, end=30, step=0.1, **plot_kwargs):
        """
        Plot the yield curve from a start to an end maturity.
        """
        if start >= end or step <= 0:
            raise ValueError("Invalid start, end, or step values.")
        if self._spline is None:
            self._fit_curve()

        plot_kwargs = plot_kwargs.copy()

        valid_plot_kwargs = {'color', 'linestyle', 'marker', 'linewidth', 'markersize'}
        for key in plot_kwargs.keys():
            if key not in valid_plot_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")

        x = np.arange(start, end + step, step)
        y = self._spline(x)
        plt.plot(x, y, **plot_kwargs)
        plt.title(plot_kwargs.get('title', 'Yield Curve'))  
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.show()
