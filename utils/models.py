"""Collection of deep learning models for market prediction."""

import torch

from torch import nn


class BasicMLP(nn.Module):
    """Basic feedforward neural network."""

    def __init__(
            self, in_dim, hidden_dim, dropout=0.0,
            activation=nn.ReLU(),
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            # Linear input layer
            nn.Linear(in_dim, hidden_dim),

            # Hidden activation 1
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,

            # # Hidden activation 2
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(dropout),
            # activation,

            # Linear output layer
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x= self.norm(x)
        x = torch.squeeze(self.mlp(x))
        return x


class EmbeddingMLP(nn.Module):
    """Basic feedforward neural network."""

    def __init__(
            self, in_dim, emb_dim, num_tokens,
            hidden_dim, dropout=0.0,
            activation=nn.ReLU(),
    ):
        super().__init__()

        input_size = in_dim + emb_dim
        self.embedding = nn.Embedding(num_tokens, emb_dim)
        # self.norm = nn.LayerNorm(in_dim)

        self.mlp = nn.Sequential(
            # Linear input layer
            nn.Linear(input_size, hidden_dim),

            # Hidden activation 1
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,

            # Hidden activation 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,

            # Linear output layer
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, i):
        # x= self.norm(x)
        embeddings = torch.squeeze(self.embedding(i))
        x = torch.cat((x, embeddings), dim=1)
        x = torch.squeeze(self.mlp(x))
        return x


# net = EmbeddingMLP(
#     in_dim=301, emb_dim=8, num_tokens=1111,
#     hidden_dim=128, dropout=0.15,
#     activation=nn.ReLU(),
# )

# input1 = torch.randn(4, 301)
# input2 = torch.randint(high=1111, size=(4, 1))
# out = net(input1, input2)
# print(out.shape)


class FeedForward(nn.Module):
    """Basic feedforward neural network."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential(
            # Linear input layer
            nn.Linear(in_dim, hidden_dim),

            # Hidden activation 1
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),

            # Hidden activation 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),

            # Linear output layer
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class AuxiliaryHead(nn.Module):
    """Regression head for auxiliary target prediction."""

    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class SingleLayerHead(nn.Module):
    """Regression head for auxiliary target prediction."""

    def __init__(self, in_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MultilayerRNN(nn.Module):
    """A combination of an MLP and RNN network."""

    def __init__(self, in_dim, hidden_dim, emb_dim, dropout=0.0):
        super().__init__()

        self.backbone = nn.Sequential(
            # Linear input layer
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),

            # Hidden activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),

            # Embedding layer
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=emb_dim, batch_first=True,
            num_layers=1, bidirectional=True, dropout=0.0,
        )

        # self.head = nn.Sequential(
        #     nn.Dropout(0.125),
        #     nn.GELU(),
        #     nn.Linear(emb_dim * 2, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        x = self.backbone(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=2)
        x = torch.sigmoid(x)
        # x = self.head(x)
        # x = torch.squeeze(x, dim=2)
        return x


class AdaptiveMLP(nn.Module):
    """A combination of an MLP and GRU network."""

    def __init__(
            self, in_dim, hidden_dim, emb_dim,
            dropout=0.0, activation=nn.ReLU()
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            # Linear input layer
            nn.Linear(in_dim, hidden_dim),
            activation,

            # Hidden activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,

            # Hidden activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,

            # Embedding layer
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
            activation,
        )

        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x - torch.mean(x, dim=1)
        x = self.linear(x)
        x = torch.squeeze(x, dim=2)
        return x


class TabTransformer(nn.Module):
    """Neural network for listwise object ranking."""

    def __init__(self, in_dim, emb_dim=64, dropout=0.0):
        super().__init__()

        # self.embedding = nn.Sequential(
        #     nn.Linear(in_dim, 128),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(128, emb_dim),
        #     nn.Dropout(dropout)
        # )

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4,
        )

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.head(x)
        return x


class PairwiseRanker(nn.Module):
    """Neural network for pairwise object ranking."""

    def __init__(self, in_dim, hidden_dim=128, emb_dim=128, dropout=0.0):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # Input layer
            nn.InstanceNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # Hidden layer
            nn.InstanceNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),  # Output layer
        )

        self.head = nn.Sequential(
            # nn.Linear(emb_dim, emb_dim // 2),  # Activation layer
            # nn.InstanceNorm1d(emb_dim // 2),
            # nn.Dropout(dropout),
            # nn.GELU(),
            # nn.Linear(emb_dim // 2, 1),  # Output node
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        features1 = self.embedding(x1)
        features2 = self.embedding(x2)
        diff = features1 - features2
        pred = self.head(diff)
        return pred
