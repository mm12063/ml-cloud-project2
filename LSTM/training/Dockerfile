# Start with a Linux micro-container to keep the image tiny
FROM debian

# Document who is responsible for this image
MAINTAINER Mitchell Merrick-Thirlway "mm12063@nyu.edu"

# Install just the Python runtime (no dev)
RUN apt-get update && apt-get install -y \
    python3 python3-dev gcc gfortran musl-dev \
    python3-pip ca-certificates

# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD requirements.txt /app
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt

# Add the code as the last Docker layer because it changes the most
ADD training.py  /app/training.py

# Run the service
CMD [ "python3", "training.py"]