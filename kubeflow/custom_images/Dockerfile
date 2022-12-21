FROM debian

RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip

COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
 && rm -f /tmp/requirements.txt