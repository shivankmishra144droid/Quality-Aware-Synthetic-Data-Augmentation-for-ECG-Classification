# Run a long GPU script with automatic restart (scripts resume from their newest checkpoint).
# Usage (from repo root, Git Bash):   bash run_retry.sh <script.py> <log file>
# Stop cleanly: create an empty file named STOP in the repo root, then end the running python
# process. The loop will not restart it. (Killing only python.exe is NOT enough: the loop restarts.)
cd "$(dirname "$0")"
rm -f STOP
for attempt in 1 2 3 4 5; do
  [ -f STOP ] && { echo "=== STOP file found, not restarting ===" >> "$2"; exit 2; }
  echo "=== attempt $attempt ===" >> "$2"
  py -3.14 -u "$1" >> "$2" 2>&1 && exit 0
  sleep 20
done
exit 1
