from faraday.llm_client import LLMClient


def test_fake_llm_satisfies_protocol(fake_llm):
    assert isinstance(fake_llm, LLMClient)        # runtime_checkable
    out = fake_llm.complete([{"role": "user", "content": "hi"}])
    assert isinstance(out, str)
