# Demo assets

Two committed, **reproducible** demo GIFs — both regenerate from code, no manual screen capture:

- **`demo-web.gif`** — the streaming web UI (README hero). The "Faraday Pi" chat answers a
  document question: the sources panel populates, the answer streams in over SSE, the citation
  lands. Regenerate on the dev machine:
  `cd scripts/web-demo && npm run setup && npm run record`
  (Playwright drives a headless browser against the Pi's `:8000` over the LAN; `ffmpeg-static`
  renders the capture to a cropped, sped-up GIF — no system ffmpeg needed).
- **`demo-cli.gif`** — the airplane-mode proof (next to the CLI quickstart). With the Pi's
  internet severed, `curl` fails on camera, yet `faraday ask` answers offline and cites its
  source. Regenerate on the Pi:
  `ssh -tt pi@raspberrypi.local 'cd ~/faraday && bash scripts/97_record_demo.sh'`
  (the wrapper drops the default route — restoring it on exit — then asciinema records the
  session and `agg` renders it).

Both pipelines use a two-pass palette, keeping each GIF far under the 5 MB hero budget
(53 K / 661 K). Tunables live in the scripts: `SPEED`/`CROP_H` env vars for the web recorder;
`stty rows/cols` and `--font-size`/`--theme` for the CLI one.
