"""Captured real tool outputs, shared by the bench parser/sweep tests."""

LLAMA_BENCH_MD = """\
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.54 B | CPU        |       4 |         pp512 |          7.71 ± 0.05 |
| qwen2 1.5B Q4_K - Medium       |   1.04 GiB |     1.54 B | CPU        |       4 |         tg128 |          3.87 ± 0.02 |

build: 1a2b3c4 (3801)
"""

TIME_V = """\
\tCommand being timed: "llama-bench -m model.gguf -p 512 -n 128"
\tUser time (seconds): 412.33
\tSystem time (seconds): 8.21
\tPercent of CPU this job got: 391%
\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:48.21
\tMaximum resident set size (kbytes): 1093284
\tExit status: 0
"""

PERPLEXITY = """\
perplexity: tokenizing the input ..
perplexity: calculating perplexity over 20 chunks
[1]6.2891,[2]7.1234,[3]6.8901,[20]6.9001,
Final estimate: PPL = 6.9543 +/- 0.08123
"""
