import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """MIL pooling

    "Attention-based Deep Multiple Instance Learning"
    (http://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf).

    Slightly modified to apply a dense layer after each attention mechanism

    Parameters
    ----------
    in_features : int
        Number of features in the input to attend.
        This should be of size dim 2 of the input (batch_size, , in_features)
    hidden_dim : int
        Number of features in the attention mechanism. Also, referred to as the
        embedding size, L, described in Section 2.4.
    V_dropout_rate : float (default=0)
        Probability of an element to be zeroed in attention mechanism V.
    U_dropout_rate : float (default=0)
        Probability of an element to be zeroed in attention mechanism U.
    gated : bool (defualt=False)
        Boolean whether to use gated attention.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        V_dropout_rate: float = 0,
        U_dropout_rate: float = 0,
        gated: bool = False,
    ):
        super().__init__()
        self._in_features = in_features
        self._hidden_dim = hidden_dim
        self._V_dropout_rate = V_dropout_rate
        self._U_dropout_rate = U_dropout_rate
        self._gated = gated

        self.attention_V = nn.Sequential(
            nn.Linear(self._in_features, self._hidden_dim, bias=False),
            nn.Tanh(),
            nn.Dropout(self._V_dropout_rate),
            nn.Linear(self._hidden_dim, 1, bias=False),
        )

        if self._gated:
            self.attention_U = nn.Sequential(
                nn.Linear(self._in_features, self._hidden_dim, bias=False),
                nn.Sigmoid(),
                nn.Dropout(self._U_dropout_rate),
                nn.Linear(self._hidden_dim, 1, bias=False),
            )
        else:
            self.attention_U = None

    def forward(self, x):
        n = x.size(1)
        x = x.reshape(-1, x.size(2))
        A = self.attention_V(x)

        if self._gated:
            A = A * self.attention_U(x)

        A = A.reshape(-1, n, 1)
        weights = F.softmax(A, dim=1)

        return (x.reshape(-1, n, self._in_features) * weights).sum(dim=1), A
