import logging

import torch

from sonar_generation import generate_from_hidden_states


class DummyTokenizer:
    lang_code_to_id = {"eng": 42}
    pad_token_id = None
    eos_token_id = 7

    def convert_tokens_to_ids(self, token: str) -> int:
        # Only the target language token is expected during the test.
        if token == "eng":
            return self.lang_code_to_id[token]
        raise KeyError(token)

    def batch_decode(self, outputs, skip_special_tokens: bool = True):
        return ["decoded"] * len(outputs)


class DummyDecoder:
    def __init__(self) -> None:
        self.last_generation_config = None
        self.config = type("Config", (), {"pad_token_id": None, "eos_token_id": None})()

    def generate(self, *, encoder_outputs, decoder_input_ids, generation_config):
        self.last_generation_config = generation_config
        return torch.tensor([[0]])


def test_generate_from_hidden_states_uses_eos_token_as_pad_when_missing(caplog):
    tokenizer = DummyTokenizer()
    decoder = DummyDecoder()
    hidden_states = torch.zeros(1, 1, 4)

    with caplog.at_level(logging.DEBUG):
        outputs = generate_from_hidden_states(
            decoder,
            tokenizer,
            hidden_states,
            "eng",
        )

    assert outputs == ["decoded"]
    assert decoder.last_generation_config.pad_token_id == tokenizer.eos_token_id
    # Confirm that the debug log helped to signal the fallback choice.
    assert any(
        "Using eos_token_id as pad_token_id" in message
        for message in caplog.messages
    )
