from faraday.eval.judge import JudgeVerdict, build_abstention_prompt, build_judge_prompt


def test_build_judge_prompt_includes_all_parts():
    p = build_judge_prompt(
        question="When did Apollo 11 land?",
        reference_answer="July 1969.",
        context="[1] Apollo 11 landed on 20 July 1969.",
        answer="It landed in July 1969 [1].",
    )
    assert "When did Apollo 11 land?" in p
    assert "July 1969." in p
    assert "Apollo 11 landed on 20 July 1969" in p
    assert "It landed in July 1969 [1]." in p
    assert "faithfulness" in p.lower() and "correctness" in p.lower()


def test_judge_verdict_holds_scores():
    v = JudgeVerdict(faithfulness=5, correctness=4, rationale="grounded; minor omission")
    assert v.faithfulness == 5 and v.correctness == 4


def test_build_abstention_prompt_includes_question_and_answer():
    p = build_abstention_prompt(
        question="What color is the LM?",
        answer="The sources do not contain this information.",
    )
    assert "What color is the LM?" in p
    assert "The sources do not contain this information." in p
    assert "abstained" in p.lower()
