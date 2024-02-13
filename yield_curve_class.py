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

        # Make a copy of plot_kwargs to avoid modifying the original
        plot_kwargs = plot_kwargs.copy()

        # Validate plot_kwargs
        valid_plot_kwargs = {'color', 'linestyle', 'marker', 'linewidth', 'markersize'}
        for key in plot_kwargs.keys():
            if key not in valid_plot_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")

        x = np.arange(start, end + step, step)
        y = self._spline(x)
        plt.plot(x, y, **plot_kwargs)
        plt.title(plot_kwargs.get('title', 'Yield Curve'))  # Use get instead of pop
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.show()


# Example maturities and rates
maturities = [1, 2, 3, 5, 7, 10, 20, 30]
rates = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.8]

# Example use
yc = YieldCurve(maturities, rates)
print(f'Rate at maturity 4: {yc.get_rate(4)}')
yc.plot_curve()
