import torch
import torch.nn as nn
from einops import rearrange
from src.layers import AttentionPool


class HRDPNet(nn.Module):
    """HR-DNN

    Parameters
    ----------
    encoder : nn.Module
        CNN encoder (backbone)
    transformer : nn.Module
        Transformer encoder
    num_features : int
        Number of features (embeddings) extracted by the encoder
    drop_rate : float
        Dropout rate
    """

    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        num_features: int,
        drop_rate: float = 0.2,
    ):
        super(HRDPNet, self).__init__()
        self.encoder = encoder
        self._num_features = num_features

        self.hr_pool = AttentionPool(self._num_features, self._num_features, gated=True)
        self.transformer = transformer

        self.global_pool = AttentionPool(
            self._num_features, self._num_features, gated=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(num_features, 1, bias=False),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module, std: float = 0.02) -> None:
        """https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        b = x.size(0)

        # encoder
        x = self.encoder(x)

        # HR-attention pooling
        x = rearrange(x, "b c h w -> b h c w")
        x, A_hr = self.hr_pool(x)
        x = rearrange(x, "(b w) c -> b w c", b=b)

        # multi-head attention
        x = self.transformer(x)

        # global pooling (attention)
        x, A = self.global_pool(x)

        # fc layer
        x = self.classifier(x)

        if return_attention:
            return x, A, A_hr
        else:
            return x
