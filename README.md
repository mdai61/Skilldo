# Player Performance Prediction using PyTorch

## Overview
This project implements a **deep learning model using PyTorch** to predict **player performance scores** and **injury risks** based on physiological, performance, and psychological metrics collected over **5 weeks**. The model utilizes **LSTM layers** to handle time-series data and learn sequential dependencies, helping coaches and analysts make data-driven decisions about player readiness and injury prevention.

## Features
### **Three Inputs (Time-Series Data)**
- **Physiological Metrics**: Includes heart rate, respiratory rate, oxygen saturation, and blood sugar levels. These metrics help assess a player's physical condition.
- **Performance Metrics**: Covers speed, agility, endurance, and other measurable performance factors to track player fitness over time.
- **Psychological and Environmental Factors**: Includes mental fatigue, stress levels, and environmental conditions to understand their impact on player well-being.

### **Two Outputs (Predictions)**
- **Performance Score (Regression Output)**: Predicts how well a player will perform based on the past five weeks of training data.
- **Injury Risk (Binary Classification)**: Determines the likelihood of a player sustaining an injury during the match.

### **Deep Learning Architecture**
- **LSTM-based Model**:
  - Each input category (Physiological, Performance, Psychological) is processed by a separate LSTM layer.
  - The outputs of these LSTMs are concatenated and passed through fully connected (Dense) layers.
  - The network generates two predictions: a performance score and an injury risk probability.
- **Loss Functions**:
  - **Mean Squared Error (MSE)** for performance score regression.
  - **Binary Cross-Entropy (BCE)** for injury risk classification.
- **Optimization**:
  - The model is trained using the **Adam optimizer** to improve convergence speed and efficiency.

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/player-performance-pytorch.git
cd player-performance-pytorch
pip install torch numpy
```

## Usage
Run the training script to train the model with simulated player data:

```bash
python train.py
```

### **Model Structure**
```
- PlayerPerformanceModel (PyTorch)
  ├── LSTM for Physiological Data
  ├── LSTM for Performance Data
  ├── LSTM for Psychological Data
  ├── Fully Connected Layer (Combining Features)
  ├── Performance Score Output (Regression)
  ├── Injury Risk Output (Classification)
```

### **Example Training Output**
```
Epoch 1, Loss: 0.5421
Epoch 2, Loss: 0.4983
...
Epoch 20, Loss: 0.3156
```

## Dataset (Simulated Data for Demonstration)
The dataset consists of **100 samples**, each containing:
- **5 time steps (weeks)**
- **Randomly generated physiological, performance, and psychological metrics**
- **Ground truth labels for performance score and injury risk**

For real-world applications, you can replace the simulated data with actual **player tracking data** from wearable sensors.

## Model Training & Evaluation
- The model is trained for **20 epochs** with batch size **32**.
- The training process updates the weights using the **Adam optimizer**.
- The loss function minimizes the difference between predicted and actual performance scores while optimizing the injury risk classification.

## Future Improvements
- **Implement GRU** as an alternative to LSTMs for better computational efficiency.
- **Use real-world datasets** with player tracking data instead of simulated values.
- **Deploy the model** as a REST API using **Flask or FastAPI** for real-time predictions.

## Deployment
- The trained model can be exported and used for inference.
- Potential deployment methods include **TorchServe, ONNX, or Flask API**.

## License
This project is licensed under the **MIT License**.

---
### **Contributions & Feedback**
We welcome contributions! If you find issues or have suggestions, feel free to open a pull request or create an issue.
