# Start with a Linux micro-container to keep the image tiny
FROM debian

# Document who is responsible for this image
MAINTAINER Mitchell Merrick-Thirlway "mm12063@nyu.edu"

# Install just the Python runtime (no dev)
RUN apt-get update && apt-get install -y \
    python3 python3-dev gcc gfortran musl-dev \
    python3-pip ca-certificates

ADD training/requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt
