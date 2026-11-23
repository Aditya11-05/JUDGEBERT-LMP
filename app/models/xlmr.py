"""XLM-R pair regressor for legal meaning-preservation scoring."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class XLMRPairRegressor(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.15, pretrained: bool = True, config_path=None):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(backbone)
        else:
            config = AutoConfig.from_pretrained(config_path or backbone, local_files_only=True)
            self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # XLM-R pair encoding uses <s> original </s></s> simplified </s>.
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state[:, 0, :]
        # Bound output to the annotation scale [1, 10].
        return 1.0 + 9.0 * torch.sigmoid(self.regressor(self.dropout(hidden)).squeeze(-1))
