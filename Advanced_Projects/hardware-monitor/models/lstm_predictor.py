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
        # Split data for validation (80/20 split)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        X_val   = torch.tensor(X_val,   dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_val   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(-1).to(self.device)

        best_val_loss = float('inf')
        for _ in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(X_train)
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss   = self.criterion(val_output, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        self.trained = True

    def predict(self, series):
        # Predict next pred_steps values
        if len(series) < self.window_size:
            raise ValueError(
                f"Series length ({len(series)}) must be at least window_size ({self.window_size})"
            )
        self.model.eval()
        input_seq = torch.tensor(
            series[-self.window_size:], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1).to(self.device)
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
