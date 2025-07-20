import torch
import torch.nn as nn
import numpy as np

class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMPredictor:
    def __init__(self, window_size=60, pred_steps=60, device=None):
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.trained = False

    def fit(self, series, epochs=10):
        # series: 1D numpy array
        X, y = self._create_dataset(series)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        for _ in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
        self.trained = True

    def predict(self, series):
        # Predict next pred_steps values
        self.model.eval()
        input_seq = torch.tensor(series[-self.window_size:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        preds = []
        seq = input_seq.clone()
        for _ in range(self.pred_steps):
            with torch.no_grad():
                out = self.model(seq)
            preds.append(out.item())
            # out shape: (1, 1), need to append to seq (1, window_size, 1)
            out_seq = out.view(1, 1, 1)
            seq = torch.cat([seq[:, 1:, :], out_seq], dim=1)
        return np.array(preds)

    def update(self, series):
        # Optionally retrain on new data
        self.fit(series, epochs=2)

    def _create_dataset(self, series):
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i+self.window_size])
            y.append(series[i+self.window_size])
        return np.array(X), np.array(y)
