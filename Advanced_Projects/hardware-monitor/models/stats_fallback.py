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
        self.model = ARIMA(series, order=self.order).fit()
        self.fitted = True

    def predict(self, series):
        if not self.fitted:
            self.fit(series)
        forecast = self.model.forecast(steps=self.pred_steps)
        return np.array(forecast)
