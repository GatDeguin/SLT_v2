from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


class _ShallowDecoder:
    """Utility to run the top decoder layers as a shallow pooling module."""

    def __init__(self, decoder, num_layers: int):
        total_layers = len(decoder.layers)
        if num_layers <= 0 or num_layers > total_layers:
            raise ValueError(
                f"num_layers must be in [1, {total_layers}] for shallow decoder, got {num_layers}"
            )
        self.decoder = decoder
        self.layer_indices = list(range(total_layers - num_layers, total_layers))

    def __call__(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        decoder = self.decoder
        inputs_embeds = decoder.embed_tokens(decoder_input_ids)
        positions = decoder.embed_positions(decoder_input_ids, inputs_embeds, past_key_values_length=0)
        positions = positions.to(inputs_embeds.device)
        hidden_states = inputs_embeds + positions
        hidden_states = F.dropout(hidden_states, p=decoder.dropout, training=decoder.training)

        if decoder_input_ids.size(1) > 1:
            seq_len = decoder_input_ids.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            mask = mask.masked_fill(mask == 1, float("-inf"))
            attention_mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = None

        cross_attention_mask: Optional[torch.Tensor] = None
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dim() == 2:
                cross_attention_mask = 1.0 - encoder_attention_mask[:, None, None, :].to(hidden_states.dtype)
            else:
                cross_attention_mask = encoder_attention_mask.to(hidden_states.dtype)
            finfo = torch.finfo(hidden_states.dtype)
            cross_attention_mask = cross_attention_mask * finfo.min

        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        for idx in self.layer_indices:
            decoder_layer = decoder.layers[idx]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attention_mask,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
            )[0]

        hidden_states = decoder.layer_norm(hidden_states)
        return hidden_states


class TextPooler:
    """Creates sentence embeddings with a shallow SONAR decoder."""

    def __init__(
        self,
        model,
        tokenizer,
        lang: str,
        *,
        num_layers: int = 4,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._lang = lang
        self._num_layers = num_layers
        self._decoder = self._resolve_decoder(model)
        self._shallow_decoder = _ShallowDecoder(self._decoder, num_layers)

        if not hasattr(tokenizer, "lang_code_to_id") or lang not in tokenizer.lang_code_to_id:
            raise ValueError(f"Language '{lang}' not available in tokenizer.lang_code_to_id")
        self._bos_id = tokenizer.lang_code_to_id[lang]

    def _resolve_decoder(self, model):
        if hasattr(model, "get_decoder"):
            return model.get_decoder()
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "get_decoder"):
            return base.get_decoder()
        raise AttributeError("Provided model does not expose a decoder")

    def _resolve_encoder(self, model):
        if hasattr(model, "get_encoder"):
            return model.get_encoder()
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "get_encoder"):
            return base.get_encoder()
        raise AttributeError("Provided model does not expose an encoder")

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("encode expects at least one text")

        device = next(self._model.parameters()).device
        self._tokenizer.src_lang = self._lang
        batch = self._tokenizer(texts, return_tensors="pt", padding=True).to(device)

        encoder = self._resolve_encoder(self._model)
        encoder_outputs = encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        labels = batch["input_ids"]
        decoder_input_ids = self._model.prepare_decoder_input_ids_from_labels(labels=labels)
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (labels.size(0), 1), self._bos_id, device=device, dtype=torch.long
            )
        else:
            decoder_input_ids = decoder_input_ids.to(device)

        decoder_hidden = self._shallow_decoder(
            decoder_input_ids,
            encoder_hidden_states,
            batch["attention_mask"],
        )
        pooled = decoder_hidden[:, 0, :]
        return pooled

    @property
    def bos_id(self) -> int:
        return self._bos_id
