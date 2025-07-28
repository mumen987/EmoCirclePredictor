# Emotional Circle Visualization with BiLSTM Prediction

## Overview

This project implements a BiLSTM-based predictive model for emotional states represented in a valence-arousal plane, visualized as an animated unit circle. The model predicts arousal, valence, and stimulation values from time-series data, clamps predictions to a unit circle, and incorporates real-time external stimulation via a Matplotlib slider UI. It supports optional UDP integration with Unity for sending predicted valence and arousal values.

The system processes CSV data containing emotional metrics, trains a bidirectional LSTM network, and generates real-time animations with quadrant-based emojis representing emotions (e.g., happy, angry, sad, excited).

Key features:
- Data preprocessing with sliding windows and MinMax normalization.
- BiLSTM model with dropout, noise injection, and early stopping.
- Interactive animation UI for real-time prediction and stimulation adjustment.
- Modular design for easy extension (e.g., Unity integration isolated).

This is built using Python, PyTorch for the model, and Matplotlib for visualization. It demonstrates time-series forecasting in affective computing, suitable for applications in human-computer interaction or emotion AI.

## Problem Analysis

The core challenge is predicting sequential emotional states (arousal, valence, stimulation) from historical data while visualizing them in a constrained 2D space (unit circle for valence-arousal). Key issues include:
- Handling temporal dependencies: BiLSTM captures bidirectional context in sequences.
- Normalization and clamping: Ensures predictions stay within valid emotional bounds.
- Real-time interaction: Slider for stimulation input, with optional external system integration (Unity).
- Overfitting prevention: Dropout, noise, and early stopping during training.
- Modularity: Separate modules for data prep, modeling, training, animation, and UDP sending to avoid tight coupling.

Theoretical background: Valence-arousal model (Russell's circumplex) maps emotions to a circle. BiLSTM excels in sequence prediction with O(seq_length * hidden^2) time complexity per forward pass.

## Requirements

- Python 3.9+
- Libraries:
  - PyTorch (>=2.0) for model definition and training.
  - NumPy, Pandas, Scikit-learn for data handling.
  - Matplotlib for animation and UI.
  - Socket (standard library) for optional UDP.

Install dependencies:
```
pip install torch numpy pandas scikit-learn matplotlib
```

Hardware: CPU sufficient; GPU recommended for faster training (use `torch.cuda.is_available()`).

Dataset: A CSV file like `prepare_data.csv` with columns: delay, arousal, valence, stimulation, sample. Example format:
```
delay,arousal,valence,stimulation,sample
0.1,0.5,0.3,0.2,1
...
```
Emoji images: Place `emoji_q1.png` to `emoji_q4.png` in an `image/` folder (e.g., for quadrants: happy, angry, sad, excited).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/emotional-circle-bilstm.git
   cd emotional-circle-bilstm
   ```

2. Install dependencies (as above).

3. Prepare data: Place your CSV in the project root or update paths in `main.py`.

4. (Optional) For Unity integration: Ensure Unity listens on UDP IP:127.0.0.1, Port:12345. Set `send_to_unity=True` in `main.py`.

## Usage

Run the main script to train the model and launch the animation:
```
python main.py
```

- Training: Loads data, trains BiLSTM, saves best model (`model.pth`), plots loss curves.
- Animation: Starts Matplotlib UI with slider for stimulation (-1 to 1). Predictions update every 200ms, clamped to circle. Emojis switch by quadrant.
- Unity: If enabled, sends valence/arousal via UDP per frame.

Customization:
- Hyperparameters: Edit in `main.py` (e.g., `seq_length=4`, `hidden_size=16`).
- Window size: Set `window_size=100` to limit trajectory history.
- Initial window: Uses first validation sequence; customize via `X_val`.

Example output: Animation shows a blue trajectory on the circle, red point for current state, emoji at position, and stimulation text.

## File Structure

- `data_preparation.py`: Loads/normalizes data, creates sequences, splits train/val.
- `model.py`: Defines BiLSTMPredictor (BiLSTM + FC layers).
- `trainer.py`: Trains model with Adam, scheduler, early stopping.
- `animation.py`: Handles Matplotlib animation, slider, clamping, emoji display. Optional Unity send.
- `unity_sender.py`: Isolated UDP sender for valence/arousal to Unity.
- `main.py`: Orchestrates data prep, training, animation. Entry point.
- `prepare_data.csv`: Sample data (not included; provide your own).
- `image/`: Folder for emoji PNGs.
- `model.pth`: Saved model weights (generated after training).

## Solutions and Implementation Details

### Model Architecture
- Input: Sequence of [delay, arousal, valence, stimulation] (shape: batch, seq_length, 4).
- BiLSTM: Bidirectional, hidden_size=16, num_layers=1, dropout=0.2. Outputs concatenated forward/backward states.
- FC: Linear(32→16) + ReLU + Dropout + Linear(16→3) for [arousal, valence, stimulation].
- Noise: Gaussian (std=0.02) added during training for robustness.
- Why BiLSTM? Captures past/future context in emotional sequences; better than vanilla LSTM for short-term dependencies.

### Training Process
- Optimizer: Adam (lr=1e-3).
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5).
- Loss: MSE.
- Early stopping: Patience=10 on val loss.
- Batch: Shuffled, size=32.
- Complexity: Time O(epochs * samples * seq_length * hidden^2), Space O(hidden * layers + batch * features).

### Animation Logic
- Initializes with validation window.
- Per frame: Get slider stimulation, predict next state, clamp (v,a) to unit circle, update window, plot trajectory/point/emoji.
- Quadrant mapping: Based on signs of valence (x) and arousal (y).
- Blitting for efficient redraw.

### Data Preprocessing
- MinMaxScaler for features/targets to [0,1].
- Sliding window: seq_length=4 for input sequences.

## Result Expectations and Validation

- Training: Expect train/val MSE <0.01 after 50-100 epochs (data-dependent). Plot shows decreasing losses.
- Animation: Smooth trajectory within circle; emojis change quadrants (e.g., +v/-a → happy).
- Validation: Compare predictions vs. ground truth on val set (MSE). Visually inspect UI for clamping (no points outside circle).
- Performance: Training ~1min on CPU for small datasets; animation ~5 FPS.
- Metrics: Track arousal/valence RMSE; aim <0.1 post-clamping.

If issues: Check data scaling (inverse_transform correct?); ensure emojis load.

## Optimization Suggestions

- Model: Increase hidden_size (32+) or layers (2) for complex data; add attention for longer sequences. Time complexity scales quadratically with hidden—profile with %timeit.
- Training: Use DataLoader for larger batches; AMP for mixed-precision on GPU to halve time. Add L1/L2 regularization if overfitting persists.
- Animation: Thread UDP sends to avoid UI lag; use faster libs like PyQt for production UI.
- Efficiency: Precompute scalers if deploying; prune model (torch.quantize) for edge devices.
- Extensions: Integrate real datasets (e.g., DEAP for emotions via web search: site:kaggle.com emotional datasets); add MLflow for tracking.
- Further: Migrate to Transformer for better long-range deps; analyze bias in quadrants via confusion matrix on predicted emotions.

For latest trends, search X or web for "BiLSTM emotion prediction 2025" or browse papers on arXiv.

## Contributing

Pull requests welcome! Focus on bug fixes, features (e.g., more emotions), or docs. Use GitHub issues for discussions.

## License

MIT License. See LICENSE file for details.