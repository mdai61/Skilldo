import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
import numpy as np

# Constants
time_steps = 5   # 5 weeks of data
physio_features = 4   # Features for physiological metrics
perf_features = 3     # Features for performance metrics
psych_features = 3    # Features for psychological metrics

class PlayerPerformanceModel(Model):
    def __init__(self, time_steps, physio_features, perf_features, psych_features):
        super(PlayerPerformanceModel, self).__init__()
        
        # LSTM layers for different inputs
        self.lstm_physio = LSTM(64, activation='tanh', return_sequences=False)
        self.lstm_perf = LSTM(64, activation='tanh', return_sequences=False)
        self.lstm_psych = LSTM(64, activation='tanh', return_sequences=False)
        
        # Fully connected layers for outputs
        self.fc_common = Dense(32, activation='relu')
        self.fc_performance = Dense(1, activation='linear')  # Regression output
        self.fc_injury = Dense(1, activation='sigmoid')  # Binary classification output
    
    def call(self, inputs):
        physio, perf, psych = inputs
        
        # Process each input through LSTM
        h_physio = self.lstm_physio(physio)
        h_perf = self.lstm_perf(perf)
        h_psych = self.lstm_psych(psych)
        
        # Concatenate hidden states
        concat = Concatenate()([h_physio, h_perf, h_psych])
        
        # Common fully connected layer
        x = self.fc_common(concat)
        
        # Outputs
        performance_score = self.fc_performance(x)
        injury_risk = self.fc_injury(x)
        
        return performance_score, injury_risk

# Initialize model
model = PlayerPerformanceModel(time_steps, physio_features, perf_features, psych_features)

# Compile the Model
model.compile(
    optimizer='adam',
    loss={'output_1': 'mse', 'output_2': 'binary_crossentropy'},
    metrics={'output_1': 'mae', 'output_2': 'accuracy'}
)

# Generate example training data
X_physio = np.random.rand(100, time_steps, physio_features)
X_perf = np.random.rand(100, time_steps, perf_features)
X_psych = np.random.rand(100, time_steps, psych_features)

y_score = np.random.rand(100, 1)  # Regression target
y_risk = np.random.randint(0, 2, size=(100, 1))  # Binary classification target

# Train the Model
model.fit(
    [X_physio, X_perf, X_psych],  # Inputs
    [y_score, y_risk],  # Outputs
    epochs=20,
    batch_size=32
)
