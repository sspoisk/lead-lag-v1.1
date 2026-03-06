#!/bin/bash
# Auto-backup to GitHub
cd /root/lead_lag_trader

# Check for changes
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    exit 0  # Nothing to backup
fi

git add -A
git commit -m "Auto-backup $(date '+%Y-%m-%d %H:%M')"
git push origin master 2>&1
