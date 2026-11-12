#!/bin/bash
# Runs the lines of queue.txt one at a time, in order (re-reading the file, so lines can be appended while it runs).
# Waits for any already running 03_vmec / 05_desc / 04_desc job first. Done lines are logged in queue.done.
cd "$(dirname "$0")"; . /home/user/venv/bin/activate; export JAX_PLATFORMS=cpu
while pgrep -f "python -u 0[345]_|python 0[345]_" >/dev/null; do sleep 10; done
touch queue.done
while true; do
  n=$(wc -l < queue.done)
  line=$(sed -n "$((n+1))p" queue.txt)
  [ -z "$line" ] && { sleep 20; line=$(sed -n "$((n+1))p" queue.txt); [ -z "$line" ] && break; }
  echo "$(date +%T) START $line" >> queue.log
  bash -c "$line"
  echo "$(date +%T) END($?) $line" >> queue.log
  echo "$line" >> queue.done
done
echo "$(date +%T) queue empty" >> queue.log
