import torch
import torch.nn as nn

class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2, noise_std=0.02):
        super(BiLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_std = noise_std

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # Weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.fc[0].weight)
        self.fc[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc[3].weight)
        self.fc[3].bias.data.fill_(0)

    def forward(self, x):
        # Add noise to input during training for robustness
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_length, hidden_size*2)
        last_step = lstm_out[:, -1, :]  # Take the last time-step as feature
        out = self.fc(last_step)  # out: (batch, output_size)
        return out