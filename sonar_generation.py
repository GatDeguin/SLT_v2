"""Shared helpers for SONAR-SLT decoding."""
from __future__ import annotations

from typing import List, Optional

import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput


def resolve_lang_token_id(tokenizer, model, lang_code: str) -> Optional[int]:
    """Best-effort resolution of language tokens across SONAR/NLLB variants."""

    lang_code_map = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_map, dict) and lang_code in lang_code_map:
        return lang_code_map[lang_code]

    config_map = getattr(getattr(model, "config", None), "lang_code_to_id", None)
    if isinstance(config_map, dict) and lang_code in config_map:
        return config_map[lang_code]

    lang_token_to_id = getattr(tokenizer, "lang_token_to_id", None)
    if callable(lang_token_to_id):
        try:
            token_id = lang_token_to_id(lang_code)
        except KeyError:
            token_id = None
        if token_id is not None:
            return token_id

    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if isinstance(token_id, int) and token_id != getattr(tokenizer, "unk_token_id", None):
        return token_id

    additional_tokens = getattr(tokenizer, "additional_special_tokens", None)
    additional_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if (
        isinstance(additional_tokens, list)
        and isinstance(additional_ids, list)
        and len(additional_tokens) == len(additional_ids)
    ):
        try:
            idx = additional_tokens.index(lang_code)
        except ValueError:
            pass
        else:
            token_id = additional_ids[idx]
            if isinstance(token_id, int):
                return token_id

    return None


def _available_lang_preview(tokenizer) -> str:
    available_langs: List[str] = []
    langs_attr = getattr(tokenizer, "langs", None)
    if isinstance(langs_attr, list):
        available_langs = list(langs_attr)
    else:
        maybe_tokens = getattr(tokenizer, "additional_special_tokens", None)
        if isinstance(maybe_tokens, list):
            available_langs = [tok for tok in maybe_tokens if isinstance(tok, str)]
    if not available_langs:
        return "unknown"
    preview = sorted(available_langs)
    return ", ".join(preview[:10])


def generate_from_hidden_states(
    decoder,
    tokenizer,
    hidden_states: torch.Tensor,
    lang_code: str,
    *,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> List[str]:
    """Decode text from encoder hidden states using SONAR/NLLB decoder."""

    forced_bos_id = resolve_lang_token_id(tokenizer, decoder, lang_code)
    if forced_bos_id is None:
        preview = _available_lang_preview(tokenizer)
        raise ValueError(
            f"Unable to resolve token id for target language '{lang_code}'. "
            f"Known languages (sample): {preview}"
        )

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        forced_bos_token_id=forced_bos_id,
        decoder_start_token_id=forced_bos_id,
        use_cache=True,
    )
    outputs = decoder.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
        decoder_input_ids=None,
        generation_config=gen_cfg,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
