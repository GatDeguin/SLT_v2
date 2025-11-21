import re

import pytest

from slt.utils.metadata import SplitSegment, parse_split_column


def test_parse_split_column_handles_unicode_double_quotes():
    segments = parse_split_column('[(“Hola”, 0, 1)]')

    assert segments == [SplitSegment("Hola", 0.0, 1.0)]


def test_parse_split_column_rejects_negative_times():
    message = "Segmento con rango temporal inválido en posición 0: ('Hola', -1, 1)"

    with pytest.raises(ValueError, match=re.escape(message)):
        parse_split_column("[('Hola', -1, 1)]")


def test_parse_split_column_rejects_non_sequential_times():
    message = "Segmento con rango temporal inválido en posición 0: ('Hola', 5, 5)"

    with pytest.raises(ValueError, match=re.escape(message)):
        parse_split_column("[('Hola', 5, 5)]")
