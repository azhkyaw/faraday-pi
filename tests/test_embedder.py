from faraday.embedder import Embedder


def test_fake_embedder_satisfies_protocol(fake_embedder):
    assert isinstance(fake_embedder, Embedder)          # runtime_checkable
    vecs = fake_embedder.embed(["hello world", "hello"])
    assert len(vecs) == 2
    assert len(vecs[0]) == fake_embedder.dim
