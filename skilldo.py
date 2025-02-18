import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Constants
time_steps = 5   # 5 weeks of data
physio_features = 4   # Features for physiological metrics
perf_features = 3     # Features for performance metrics
psych_features = 3    # Features for psychological metrics

class PlayerPerformanceModel(nn.Module):
    def __init__(self, time_steps, physio_features, perf_features, psych_features):
        super(PlayerPerformanceModel, self).__init__()
        
        # LSTM layers for different inputs
        self.lstm_physio = nn.LSTM(input_size=physio_features, hidden_size=64, batch_first=True)
        self.lstm_perf = nn.LSTM(input_size=perf_features, hidden_size=64, batch_first=True)
        self.lstm_psych = nn.LSTM(input_size=psych_features, hidden_size=64, batch_first=True)
        
        # Fully connected layers for outputs
        self.fc_common = nn.Linear(64 * 3, 32)
        self.fc_performance = nn.Linear(32, 1)  # Regression output
        self.fc_injury = nn.Linear(32, 1)  # Binary classification output
        
    def forward(self, physio, perf, psych):
        _, (h_physio, _) = self.lstm_physio(physio)
        _, (h_perf, _) = self.lstm_perf(perf)
        _, (h_psych, _) = self.lstm_psych(psych)
        
        # Concatenate hidden states
        concat = torch.cat((h_physio[-1], h_perf[-1], h_psych[-1]), dim=1)
        
        # Common fully connected layer
        x = torch.relu(self.fc_common(concat))
        
        # Outputs
        performance_score = self.fc_performance(x)
        injury_risk = torch.sigmoid(self.fc_injury(x))
        
        return performance_score, injury_risk

# Initialize model
model = PlayerPerformanceModel(time_steps, physio_features, perf_features, psych_features)

# Loss and optimizer
criterion_performance = nn.MSELoss()
criterion_injury = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training data
X_physio = torch.rand(100, time_steps, physio_features)
X_perf = torch.rand(100, time_steps, perf_features)
X_psych = torch.rand(100, time_steps, psych_features)

y_score = torch.rand(100, 1)  # Regression target
y_risk = torch.randint(0, 2, size=(100, 1)).float()  # Binary classification target

# Training loop
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    pred_score, pred_risk = model(X_physio, X_perf, X_psych)
    loss_score = criterion_performance(pred_score, y_score)
    loss_risk = criterion_injury(pred_risk, y_risk)
    loss = loss_score + loss_risk
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
