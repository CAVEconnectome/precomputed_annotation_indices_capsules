FROM python:3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget bash procps \
    && rm -rf /var/lib/apt/lists/*

# Clone capsule code into image at build time
RUN git clone https://github.com/CAVEconnectome/precomputed_annotation_indices_capsules.git /capsule-repo

COPY code/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
