"""Shared helpers for SONAR-SLT decoding."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput


logger = logging.getLogger(__name__)


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

    candidate_tokens = [lang_code]
    if lang_code and not lang_code.startswith("__"):
        candidate_tokens.append(f"__{lang_code}__")

    for candidate in candidate_tokens:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
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


def _resolve_special_token_id(tokenizer, model, attr_name: str, fallback: int) -> int:
    for source in (tokenizer, getattr(model, "config", None), model):
        if source is None:
            continue
        token_id = getattr(source, attr_name, None)
        if isinstance(token_id, int):
            return token_id
        if isinstance(token_id, (list, tuple)) and token_id:
            first = token_id[0]
            if isinstance(first, int):
                return first
    return fallback


DEFAULT_GENERATION_OPTIONS: Dict[str, Any] = {
    "max_new_tokens": 64,
    "num_beams": 4,
    "do_sample": False,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
}

ALLOWED_GENERATION_OPTION_KEYS = frozenset(DEFAULT_GENERATION_OPTIONS.keys())


def _build_generation_config_kwargs(
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config_kwargs = dict(DEFAULT_GENERATION_OPTIONS)
    if overrides:
        filtered = {
            k: v
            for k, v in overrides.items()
            if k in ALLOWED_GENERATION_OPTION_KEYS and v is not None
        }
        config_kwargs.update(filtered)
    return config_kwargs


def generate_from_hidden_states(
    decoder,
    tokenizer,
    hidden_states: torch.Tensor,
    lang_code: str,
    *,
    generation_options: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Decode text from encoder hidden states using SONAR/NLLB decoder."""

    forced_bos_id = resolve_lang_token_id(tokenizer, decoder, lang_code)
    if forced_bos_id is None:
        preview = _available_lang_preview(tokenizer)
        raise ValueError(
            f"Unable to resolve token id for target language '{lang_code}'. "
            f"Known languages (sample): {preview}"
        )

    config_kwargs = _build_generation_config_kwargs(generation_options)
    gen_cfg = GenerationConfig(
        **config_kwargs,
        forced_bos_token_id=forced_bos_id,
        decoder_start_token_id=forced_bos_id,
        pad_token_id=_resolve_special_token_id(tokenizer, decoder, "pad_token_id", forced_bos_id),
        eos_token_id=_resolve_special_token_id(tokenizer, decoder, "eos_token_id", forced_bos_id),
        use_cache=True,
    )
    if logger.isEnabledFor(logging.INFO):
        generation_log = {
            "bos_id": forced_bos_id,
            "pad_id": gen_cfg.pad_token_id,
            "eos_id": gen_cfg.eos_token_id,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "no_repeat_ngram_size": gen_cfg.no_repeat_ngram_size,
            "do_sample": gen_cfg.do_sample,
            "num_beams": gen_cfg.num_beams,
            "max_new_tokens": gen_cfg.max_new_tokens,
            "length_penalty": gen_cfg.length_penalty,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
        }
        logger.info(
            "decoder.generate parameters",
            extra={"generation_parameters": generation_log},
        )
    outputs = decoder.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=hidden_states),
        decoder_input_ids=None,
        generation_config=gen_cfg,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
