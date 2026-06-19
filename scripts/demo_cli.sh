#!/usr/bin/env bash
# Recorded CLI demo sequence — run INSIDE `asciinema rec -c` by scripts/97_record_demo.sh.
# Prints a Pi-style prompt before each command (so the GIF reads as a live session), then
# runs the real commands: curl fails (internet severed at the firewall), yet `faraday ask`
# answers with a citation — generation + embeddings run entirely on this Pi.
cd ~/faraday && source .venv/bin/activate
P='\033[36mpi@raspberrypi\033[0m:\033[34m~/faraday\033[0m$ '
sleep 0.6
printf "%b%s\n" "$P" "# internet severed — prove it:"; sleep 0.7
printf "%b%s\n" "$P" 'curl --max-time 5 https://example.com'; sleep 0.3
curl --max-time 5 https://example.com || true
sleep 0.9
printf "%b%s\n" "$P" "# no internet — yet Faraday answers (gen + embed run on THIS Pi):"; sleep 0.7
printf "%b%s\n" "$P" 'faraday ask "What CPU does the Raspberry Pi 4 use?"'; sleep 0.3
faraday ask "What CPU does the Raspberry Pi 4 use?"
sleep 1.8
