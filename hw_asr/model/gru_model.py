from torch import nn
from hw_asr.base import BaseModel
from .baseline_model import BaselineModel


class GRUBlock(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=3,
                 dropout=0.1,
                 ):
        super(GRUBlock, self).__init__()
        self.lstm = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           bidirectional=True,
                           num_layers=num_layers,
                           dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class TwoLayerGRU(BaselineModel):
    def __init__(self, n_feats, n_class, lstm_hidden=512, **kwargs):
        super().__init__(n_feats, n_class, **kwargs)

        self.net = nn.Sequential(
            GRUBlock(input_size=n_feats, hidden_size=lstm_hidden),
            nn.Linear(in_features=lstm_hidden * 2, out_features=n_class)
        )
