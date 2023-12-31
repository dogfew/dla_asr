import torch
from torch import nn
from hw_asr.base import BaseModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        kernel_size: tuple[int, int] = (21, 11),
        stride: tuple[int, int] = (2, 1),
    ):
        super().__init__()
        self.padding = kernel_size[0] // 2, kernel_size[-1] // 2
        self.time_scaling = stride[1]
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.padding,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.Hardtanh(min_val=0, max_val=20, inplace=True),
            ]
        )

    def forward(self, x, lengths):
        """
        :param x: B x C x F x T
        :param lengths: B
        """
        for layer in self.layers:
            x = layer(x)
            B, C, F, T = x.shape
            mask = torch.arange(T, device=x.device).expand(B, C, F, T) >= lengths.view(
                B, 1, 1, 1
            ).to(x.device)
            x.masked_fill_(mask, 0)
        return x, lengths // self.time_scaling


class RNNBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.use_batch_norm = batch_norm

    def forward(self, x, lengths, h=None):
        """
        :param x: B x T x (C * F)
        :param lengths: B
        :param h:
        :returns : B x T x (C * F)
        """
        if self.use_batch_norm:
            B, T, F = x.shape
            x = self.batch_norm(x.view(B, F, T)).view(B, T, F)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, None)
        x, _ = pad_packed_sequence(x, batch_first=True)
        if self.rnn.bidirectional:
            x = x[:, :, : self.rnn.hidden_size] + x[:, :, self.rnn.hidden_size :]
        return x, lengths, h


class DeepSpeech2(BaseModel):
    def __init__(
        self,
        n_feats: int,
        n_class: int,
        n_conv_layers: int = 2,
        n_rnn_layers: int = 3,
        fc_hidden: int = 512,
        **batch
    ):
        assert n_feats > 0
        assert n_class > 0
        assert 0 < n_conv_layers < 4
        assert n_rnn_layers > 1
        super().__init__(n_feats, n_class, **batch)
        raw_conv_layers = [
            ConvBlock(
                in_channels=1, out_channels=32, stride=(2, 2), kernel_size=(41, 11)
            ),
            ConvBlock(in_channels=32, out_channels=32),
            ConvBlock(in_channels=32, out_channels=96),
        ]
        rnn_input_size = (
            (96 if n_conv_layers == 3 else 32) * n_feats // 2**n_conv_layers
        )
        raw_rnn_layers = [
            RNNBlock(
                input_size=rnn_input_size,
                hidden_size=fc_hidden,
                bidirectional=True,
                batch_norm=False,
            )
        ] + [
            RNNBlock(
                input_size=fc_hidden,
                hidden_size=fc_hidden,
                bidirectional=True,
                batch_norm=True,
            )
            for _ in range(n_rnn_layers - 1)
        ]

        self.conv_layers = nn.ModuleList(raw_conv_layers[:n_conv_layers])
        self.rnn_layers = nn.ModuleList(raw_rnn_layers[:n_rnn_layers])
        self.batch_norm = nn.BatchNorm1d(fc_hidden)
        self.linear = nn.Linear(fc_hidden, n_class, bias=False)

    def forward(self, spectrogram, **batch):
        """
        :param spectrogram: B x F x T
        """
        lengths = batch["spectrogram_length"]
        out = spectrogram.unsqueeze(dim=1)
        for layer in self.conv_layers:
            out, lengths = layer(out, lengths)
        B, T, F, _ = out.shape
        out = out.view(B, T * F, -1).transpose(1, 2).contiguous()
        h = None
        for layer in self.rnn_layers:
            out, lengths, h = layer(out, lengths, h)
        B, T, F = out.shape
        out = self.batch_norm(out.view(B, F, T)).view(B, T, F)
        out = self.linear(out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
