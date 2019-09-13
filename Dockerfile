# Install the nightly version of tensorflow
FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3
WORKDIR /root

# Installs pandas, google-cloud-storage, and scikit-learn
# scikit-learn is used when loading the data
RUN pip3 install pandas google-cloud-storage polarTransform

# Install curl
RUN apt-get update; apt-get install curl -y

# The data for this sample has been publicly hosted on a GCS bucket.
# Download the data from the public Google Cloud Storage bucket for this sample

# Copies the trainer code to the docker image.
COPY trainer ./trainer
RUN mkdir trained_models

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-m", "trainer.graybox_task"]
