#!/usr/bin/env bash
# Run ON the Raspberry Pi (Pi OS Lite 64-bit, Debian 13 "trixie").
# Installs the build + runtime toolchain for Faraday.
#
# Bring-up order (the steps before this script):
#   1. Flash Raspberry Pi OS Lite (64-bit) with SSH + your user preconfigured.
#   2. Add your dev machine's SSH public key to ~/.ssh/authorized_keys.
#   3. Enable passwordless sudo (so automation can install packages):
#        echo 'pi ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/010-faraday-nopasswd
#        sudo chmod 440 /etc/sudoers.d/010-faraday-nopasswd
#   4. Run this script.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  python3 python3-venv python3-pip \
  libcurl4-openssl-dev

echo "--- toolchain ---"
git --version
cmake --version | head -1
python3 --version
echo "--- hardware ---"
echo "cores: $(nproc)"
free -h
vcgencmd measure_temp

# Note: Pi OS Lite already provides a 2 GB dphys swapfile, which is sufficient
# for M0-M1 (1.5B model). zram is deferred to the M4 quantization sweep, where
# larger models put real pressure on the 4 GB ceiling.
