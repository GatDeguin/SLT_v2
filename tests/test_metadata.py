from slt.utils.metadata import SplitSegment, parse_split_column


def test_parse_split_column_handles_unicode_double_quotes():
    segments = parse_split_column('[(“Hola”, 0, 1)]')

    assert segments == [SplitSegment("Hola", 0.0, 1.0)]
