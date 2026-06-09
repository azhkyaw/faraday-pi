"""Shared fixtures for the M4b eval-engine tests (no Pi, no network)."""

# A tiny golden set: one answerable (with a labeled span), one unanswerable.
GOLDEN_JSONL = (
    '{"id":"q1","question":"When did Apollo 11 land?","answerable":true,'
    '"relevant_doc":"moon.txt","relevant_span":[100,260],'
    '"reference_answer":"July 1969."}\n'
    '{"id":"q2","question":"What is the capital of Mars?","answerable":false,'
    '"relevant_doc":"","relevant_span":[0,0],"reference_answer":""}\n'
)
