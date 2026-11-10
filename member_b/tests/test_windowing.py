from windowing import split_into_windows


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


def test_long_text_is_split_with_overlap():
    tokens = "one two three four five six seven eight nine ten"
    windows = split_into_windows(tokens, FakeTokenizer(), window_size=4, overlap=1)
    assert len(windows) >= 3
    assert windows[0]["text"].split()[-1] == windows[1]["text"].split()[0]


def test_empty_text_returns_no_windows():
    assert split_into_windows("   ", FakeTokenizer()) == []
