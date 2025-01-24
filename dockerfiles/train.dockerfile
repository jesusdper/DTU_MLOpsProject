# Base image
FROM python:3.11-slim AS base

ARG PROJECT_ID=dtumlops-448008

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && \
    apt-get install google-cloud-cli -y

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src src/
COPY configs /configs
COPY .dvc .dvc
COPY data/ data/

WORKDIR /

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --verbose --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/pip pip install . --no-deps --verbose --no-cache-dir

# Set default values for environment variables
ENV SAVE_LOCATION="/models"
ENV N_EPOCHS=1

# Set gcloud project
RUN gcloud config set project ${PROJECT_ID}


ENTRYPOINT ["python", "-u", "src/object_detection/train.py"]
CMD ["--save_location", "/models", "--n_epochs", "1"]
