#!/bin/sh
set -e
python /app/startup.py
exec supervisord -c /app/docker/supervisord.conf