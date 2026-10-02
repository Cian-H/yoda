"""Phase 1 input encoders for text, symbolic states, and constraints."""

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Encodes text sequences into continuous vector representations using HF or dummy weights."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        target_dim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = device
        self.dtype = dtype

        if model_name == "dummy":
            # Fast path for unit tests without network or heavy models
            self.tokenizer = None
            self.model = None
            self._dim = target_dim or 256
            self.dummy_embed = nn.Embedding(1000, self._dim, device=device, dtype=dtype)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device=device, dtype=dtype)
            self._dim = self.model.config.hidden_size

        # Optional adapter if target_dim is provided and differs from native hidden size
        if target_dim and target_dim != self._dim and model_name != "dummy":
            self.adapter = nn.Linear(self._dim, target_dim, device=device, dtype=dtype)
        else:
            self.adapter = nn.Identity()

    @property
    def dim(self) -> int:
        """Returns the effective dimension of output embeddings."""
        return self.target_dim or self._dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Returns sequence embeddings of shape (batch_size, seq_len, target_dim)."""
        if self.model_name == "dummy":
            batch_size = len(texts)
            dim = self.target_dim or self._dim
            dev = self.device or (
                self.dummy_embed.weight.device if hasattr(self, "dummy_embed") else None
            )
            dtype = self.dtype or (
                self.dummy_embed.weight.dtype if hasattr(self, "dummy_embed") else torch.float32
            )

            if batch_size == 0:
                return torch.empty((0, 0, dim), device=dev, dtype=dtype)

            seq_len = max((len(t.split()) for t in texts), default=0) or 1
            fake_ids = torch.randint(0, 1000, (batch_size, seq_len), device=dev)
            return self.dummy_embed(fake_ids)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        if self.model is not None:
            target_device = next(self.model.parameters()).device
            inputs = inputs.to(target_device)
        elif self.device is not None:
            inputs = inputs.to(self.device)

        outputs = self.model(**inputs)
        # Sequence of hidden states (batch_size, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        return self.adapter(hidden_states)


class SymbolicStateEncoder(nn.Module):
    """Encodes batches of symbolic dictionary states into continuous tensor sequences."""

    def __init__(self, text_encoder: TextEncoder) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def _format_dict(self, state: dict[str, Any]) -> str:
        """Flattens a dictionary into a structured string."""
        if not state:
            return "empty"
        return " | ".join(f"{k}: {v}" for k, v in state.items())

    def forward(self, states: list[dict[str, Any]]) -> torch.Tensor:
        """Encodes a batch of dictionaries into tensor sequences."""
        formatted = [self._format_dict(s) for s in states]
        return self.text_encoder(formatted)


class ConstraintEncoder(nn.Module):
    """Encodes batches of symbolic constraints into continuous tensor sequences."""

    def __init__(self, text_encoder: TextEncoder) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def _format_constraints(self, constraints: list[str]) -> str:
        """Flattens a list of constraints into a structured string."""
        if not constraints:
            return "none"
        return " || ".join(constraints)

    def encode_candidates(self, constraints_batch: list[list[str]]) -> torch.Tensor:
        """Encodes each candidate criterion individually into candidate embeddings.

        Args:
            constraints_batch: Batch of constraint/choice lists of shape
                `[batch_size, num_candidates]`.

        Returns:
            Candidate embedding tensor of shape (batch_size, num_candidates, target_dim).
        """
        batch_size = len(constraints_batch)
        if batch_size == 0:
            dim = self.text_encoder.dim
            device = self.text_encoder.device or torch.device("cpu")
            return torch.empty((0, 0, dim), device=device)

        max_candidates = max((len(s) for s in constraints_batch), default=0)
        if max_candidates == 0:
            dim = self.text_encoder.dim
            device = self.text_encoder.device or torch.device("cpu")
            return torch.empty((batch_size, 0, dim), device=device)

        padded_batch = [
            list(s) + ["none: Unused option"] * (max_candidates - len(s)) for s in constraints_batch
        ]

        flat_candidates = [c for sample in padded_batch for c in sample]
        cand_seq = self.text_encoder(flat_candidates)
        cand_pooled = cand_seq.mean(dim=1)
        return cand_pooled.view(batch_size, max_candidates, -1)

    def forward(self, constraints_batch: list[list[str]]) -> torch.Tensor:
        """Encodes batches of constraints by joining them with a separator."""
        formatted = [self._format_constraints(c) for c in constraints_batch]
        return self.text_encoder(formatted)
