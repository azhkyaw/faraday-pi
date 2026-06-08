# Sync this repo to the Raspberry Pi by pushing the current git branch.
#
# The Pi hosts a git repo at ~/faraday configured with
#   receive.denyCurrentBranch = updateInstead
# so its *working tree* updates automatically on every push. This pins the Pi
# to an exact commit (reproducible) instead of an ad-hoc file copy.
#
# One-time setup (already done during bring-up):
#   ssh pi@raspberrypi.local "mkdir -p ~/faraday && cd ~/faraday && \
#     git init -b m0-m1-rag-core -q && git config receive.denyCurrentBranch updateInstead"
#   git remote add pi pi@raspberrypi.local:faraday
#
# Usage:  ./scripts/sync.ps1
$ErrorActionPreference = "Stop"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
git push pi $branch
Write-Host "Pushed '$branch' -> Pi (~/faraday working tree updated)."
