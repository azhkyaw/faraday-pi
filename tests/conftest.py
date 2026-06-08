import pytest


class FakeEmbedder:
    """Deterministic vocab-based embeddings: each distinct word gets a stable slot,
    so the same instance embeds corpus and query consistently (no hash randomization)."""
    dim = 32

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _slot(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) % self.dim
        return self._vocab[word]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[self._slot(w)] += 1.0
            out.append(v)
        return out


class FakeLLM:
    def __init__(self, reply: str = "Answer [1]."):
        self.reply = reply
        self.last_messages = None

    def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        self.last_messages = messages
        return self.reply

    def stream(self, messages: list[dict], max_tokens: int = 512):
        self.last_messages = messages
        mid = len(self.reply) // 2          # two chunks that rejoin to the exact reply
        yield self.reply[:mid]
        yield self.reply[mid:]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def make_llm():
    """Factory for a FakeLLM with a custom canned reply."""
    return lambda reply: FakeLLM(reply)
