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


def test_http_embedder_clips_overlong_inputs():
    # bge-small-en-v1.5 caps at 512 tokens; llama-server returns a hard 500
    # ("input is too large to process") on a longer input rather than truncating,
    # which crashed the chunk_size=2400 eval ingest. The client must clip each
    # input to max_input_chars BEFORE the POST. Stored chunk text is untouched
    # (ingest passes c.text to the store separately) — only the embedded view is
    # clipped, and every input still yields exactly one vector, in order.
    sent_lengths: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        sent_lengths.extend(len(t) for t in texts)
        data = [{"index": i, "embedding": [1.0]} for i, _ in enumerate(texts)]
        return httpx.Response(200, json={"data": data})

    emb = HttpEmbedder(batch_size=16, max_input_chars=1800)
    emb._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    vecs = emb.embed(["x" * 5000, "short"])
    assert sent_lengths == [1800, 5]      # over-long input clipped, short one untouched
    assert len(vecs) == 2                 # one vector per input, order preserved
