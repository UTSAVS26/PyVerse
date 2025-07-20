import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class StatsFallbackPredictor:
    def __init__(self, order=(2, 1, 2), pred_steps=60):
        self.order = order
        self.pred_steps = pred_steps
        self.model = None
        self.fitted = False

    def fit(self, series):
        # series: 1D numpy array
        if len(series) < max(self.order) + 10:  # Minimum data requirement
            raise ValueError(
                f"Insufficient data for ARIMA({self.order}). "
                f"Need at least {max(self.order) + 10} points, got {len(series)}"
            )
        
        try:
            self.model = ARIMA(series, order=self.order).fit()
            self.fitted = True
        except Exception as e:
            raise RuntimeError(f"Failed to fit ARIMA model: {e}")

    def predict(self, series):
        if not self.fitted:
            self.fit(series)
        forecast = self.model.forecast(steps=self.pred_steps)
        return np.array(forecast)
