"""Multitask CNN classifier with pluggable sequence aggregators.

Implements the same ``{class_name: (B, 2)}`` logits-dict interface as
``MultiTaskWav2VecClassifier`` so evaluation, checkpoint loading and the
registry treat both model families identically.
"""

import torch
import torch.nn as nn

from model.config.defaults import DYSFLUENCY_CLASSES

ENCODER_CHANNELS = 128


class _PoolAggregator(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)

    def forward(self, x):
        x = x.mean(dim=(2, 3))
        return self.proj(x)


class _LSTMAggregator(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class _TransformerAggregator(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        nhead = max(1, hidden_dim // 32)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.encoder(x)
        return x.mean(dim=1)


class _CNNMultitaskBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(d),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(d),
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(d),
            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(d),
        )
        aggregator = config.aggregator
        if aggregator == 'pool':
            self.aggregator = _PoolAggregator(ENCODER_CHANNELS, config.hidden_dim)
        elif aggregator == 'lstm':
            self.aggregator = _LSTMAggregator(
                ENCODER_CHANNELS, config.hidden_dim, config.num_lstm_layers, d,
            )
        elif aggregator == 'transformer':
            self.aggregator = _TransformerAggregator(
                ENCODER_CHANNELS, config.hidden_dim, config.num_transformer_layers, d,
            )
        else:
            raise ValueError(f'Unknown aggregator: {aggregator}')
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, 2),
            )
            for name in config.class_names
        })

    def forward(self, spectrograms):
        x = self.encoder(spectrograms)
        pooled = self.aggregator(x)
        return {name: head(pooled) for name, head in self.heads.items()}


class CNNMultitaskClassifier:
    def __init__(self, n_mels=128, hop_length=512, in_channels=1, hidden_dim=128,
                 dropout=0.4, class_names=None, aggregator='pool',
                 num_lstm_layers=1, num_transformer_layers=1):
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.class_names = list(class_names) if class_names else list(DYSFLUENCY_CLASSES)
        self.aggregator = aggregator
        self.num_lstm_layers = num_lstm_layers
        self.num_transformer_layers = num_transformer_layers
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._build_model()
        return self._model

    def _build_model(self):
        self._model = _CNNMultitaskBackbone(self)

    def forward(self, spectrograms):
        return self.model(spectrograms)

    def forward_head(self, class_name, spectrograms):
        return self.model(spectrograms)[class_name]

    def predict(self, spectrogram_tensor, threshold=0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f'threshold must be in [0, 1], got {threshold}')
        if spectrogram_tensor.dim() == 3:
            spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(spectrogram_tensor)
        results = {}
        for name in self.class_names:
            probs = torch.softmax(logits[name], dim=-1)
            prob_present = probs[0, 1].item()
            label = 1 if prob_present >= threshold else 0
            confidence = prob_present if label == 1 else 1 - prob_present
            results[name] = (label, confidence)
        return results

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_mels': self.n_mels,
            'hop_length': self.hop_length,
            'in_channels': self.in_channels,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'class_names': self.class_names,
            'aggregator': self.aggregator,
            'num_lstm_layers': self.num_lstm_layers,
            'num_transformer_layers': self.num_transformer_layers,
        }, path)

    @classmethod
    def from_pretrained(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        instance = cls(
            n_mels=checkpoint['n_mels'],
            hop_length=checkpoint.get('hop_length', 512),
            in_channels=checkpoint.get('in_channels', 1),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            dropout=checkpoint.get('dropout', 0.4),
            class_names=checkpoint.get('class_names'),
            aggregator=checkpoint.get('aggregator', 'pool'),
            num_lstm_layers=checkpoint.get('num_lstm_layers', 1),
            num_transformer_layers=checkpoint.get('num_transformer_layers', 1),
        )
        state_dict = {k.replace('_orig_mod.', ''): v
                      for k, v in checkpoint['model_state_dict'].items()}
        instance.model.load_state_dict(state_dict, strict=True)
        return instance

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self):
        return (f'CNNMultitaskClassifier(n_mels={self.n_mels}, '
                f'hidden_dim={self.hidden_dim}, aggregator={self.aggregator}, '
                f'class_names={self.class_names})')
