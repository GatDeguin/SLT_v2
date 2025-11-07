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

        self._bos_id = self._resolve_bos_id(tokenizer, lang)

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

    def _resolve_bos_id(self, tokenizer, lang: str) -> int:
        bos_id: Optional[int] = None

        if hasattr(tokenizer, "lang_code_to_id"):
            bos_id = tokenizer.lang_code_to_id.get(lang)
        if bos_id is None and hasattr(tokenizer, "lang2id"):
            bos_id = tokenizer.lang2id.get(lang)
        if bos_id is None and hasattr(tokenizer, "get_lang_id"):
            try:
                bos_id = tokenizer.get_lang_id(lang)
            except (KeyError, ValueError):
                bos_id = None
        if bos_id is None:
            token_id = tokenizer.convert_tokens_to_ids(lang)
            if token_id != tokenizer.unk_token_id:
                bos_id = token_id

        if bos_id is None:
            raise ValueError(
                "Language '{lang}' not available in tokenizer.".format(lang=lang)
            )

        return bos_id

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
        decoder_input_ids = self._prepare_decoder_input_ids(labels, device)

        decoder_hidden = self._shallow_decoder(
            decoder_input_ids,
            encoder_hidden_states,
            batch["attention_mask"],
        )
        pooled = decoder_hidden[:, 0, :]
        return pooled

    def _prepare_decoder_input_ids(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        prepare_fn = self._find_prepare_decoder_input_ids_from_labels()
        if prepare_fn is not None:
            decoder_input_ids = prepare_fn(labels=labels)
            if decoder_input_ids is not None:
                return decoder_input_ids.to(device)

        return self._shift_right(labels.to(device))

    def _find_prepare_decoder_input_ids_from_labels(self):
        candidates = [self._model]

        base = getattr(self._model, "base_model", None)
        if base is not None:
            candidates.append(base)
            inner = getattr(base, "model", None)
            if inner is not None:
                candidates.append(inner)

        for candidate in candidates:
            if candidate is None:
                continue
            prepare_fn = getattr(candidate, "prepare_decoder_input_ids_from_labels", None)
            if prepare_fn is not None:
                return prepare_fn
        return None

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for shifting")

        shifted = input_ids.new_full(input_ids.shape, pad_token_id)
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[:, 0] = self._bos_id
        shifted = shifted.masked_fill(shifted == -100, pad_token_id)
        return shifted

    @property
    def bos_id(self) -> int:
        return self._bos_id
