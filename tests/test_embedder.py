import json

import httpx

from faraday.embedder import Embedder, HttpEmbedder


def test_fake_embedder_satisfies_protocol(fake_embedder):
    assert isinstance(fake_embedder, Embedder)          # runtime_checkable
    vecs = fake_embedder.embed(["hello world", "hello"])
    assert len(vecs) == 2
    assert len(vecs[0]) == fake_embedder.dim


def test_http_embedder_batches_and_preserves_order():
    # llama-server returns nothing until a whole batch is embedded, so one
    # unbounded POST per document = read-timeout on big documents. The HTTP
    # impl must bound each request and stitch results back in input order.
    request_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        request_sizes.append(len(texts))
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(texts)]
        return httpx.Response(200, json={"data": list(reversed(data))})

    emb = HttpEmbedder(batch_size=16)
    emb._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    texts = ["t" + "x" * i for i in range(40)]      # distinct lengths -> distinct vectors
    vecs = emb.embed(texts)
    assert request_sizes == [16, 16, 8]
    assert vecs == [[float(len(t))] for t in texts]
