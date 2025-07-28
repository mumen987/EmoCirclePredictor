import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model: nn.Module,
                X_train: torch.Tensor,
                Y_train: torch.Tensor,
                X_val: torch.Tensor,
                Y_val: torch.Tensor,
                num_epochs=100,
                batch_size=32,
                learning_rate=1e-3,
                patience=10,
                device=None,
                model_save_path=None):
    """
    Train the BiLSTMPredictor with early stopping.
    Use Adam optimizer, ReduceLROnPlateau scheduler, and MSE loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    def get_batches(X, Y, batch_size):
        perm = torch.randperm(X.size(0))
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]
        for i in range(0, X_shuffled.size(0), batch_size):
            yield X_shuffled[i: i + batch_size], Y_shuffled[i: i + batch_size]

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in get_batches(X_train, Y_train, batch_size):
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        epoch_train_loss = running_loss / X_train.size(0)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for i in range(0, X_val.size(0), batch_size):
                xb = X_val[i: i + batch_size]
                yb = Y_val[i: i + batch_size]
                preds = model(xb)
                val_loss_sum += criterion(preds, yb).item() * xb.size(0)
            epoch_val_loss = val_loss_sum / X_val.size(0)
            val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            if model_save_path:
                torch.save(model.state_dict(), str(model_save_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation loss has not improved for {patience} epochs. Early stopping.")
                break

    return train_losses, val_losses