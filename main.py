from pathlib import Path
import torch
import matplotlib.pyplot as plt

from data_preparation import prepare_data
from model import BiLSTMPredictor
from trainer import train_model
from animation import animate_with_stimulation_ui

if __name__ == "__main__":
    base_dir = Path(__file__).parent

    data_path = base_dir / "prepare_data.csv"
    model_save_path = base_dir / "model.pth"
    emoji_folder = base_dir / "image"  # Ensure emoji_q1.png~emoji_q4.png exist

    # Hyperparameters
    seq_length = 4
    hidden_size = 16
    num_layers = 1
    dropout = 0.2
    noise_std = 0.02
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-3
    patience = 10

    # Prepare data
    X_train, Y_train, X_val, Y_val, scaler_Y, scaler_X = prepare_data(
        data_path, seq_length=seq_length, test_size=0.2
    )

    # Build model
    input_size = 4  # delay, arousal, valence, stimulation
    output_size = 3  # arousal, valence, stimulation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPredictor(input_size, hidden_size, output_size,
                            num_layers=num_layers, dropout=dropout, noise_std=noise_std)

    # Train model
    train_losses, val_losses = train_model(
        model,
        X_train, Y_train,
        X_val, Y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        model_save_path=model_save_path
    )

    # Visualize training/validation loss curves (optional)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.tight_layout()
    plt.show()

    # Load best model weights
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # Initial window from validation set
    init_window = X_val[0].unsqueeze(0).to(device)  # (1, seq_length, 4), normalized

    # Start animation (set send_to_unity=True to enable Unity sending)
    animate_with_stimulation_ui(
        model=model,
        init_window_norm=init_window,
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        emoji_folder=emoji_folder,
        device=device,
        interval=200,
        window_size=100,  # Show recent 100 frames in trajectory
        send_to_unity=False  # Change to True if Unity integration needed
    )