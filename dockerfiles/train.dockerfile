# Base image
FROM python:3.11-slim AS base

ARG PROJECT_ID

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src src/
COPY configs /configs
COPY .dvc .dvc
COPY dockerfiles/entrypoint.sh entrypoint.sh
# Add executable permission to file
RUN chmod +x /entrypoint.sh

WORKDIR /

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --verbose --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/pip pip install . --no-deps --verbose --no-cache-dir

# Set default values for environment variables
ENV SAVE_LOCATION="/models"
ENV N_EPOCHS=1

# Set gcloud project
RUN gcloud config set project ${PROJECT_ID}

# Download data from GCS
RUN gsutil -m cp -r gs://pascalvoc_mlops/data /data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-u", "src/object_detection/train.py", "--save_location", "$SAVE_LOCATION", "--n_epochs", "$N_EPOCHS"]