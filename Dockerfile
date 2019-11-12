# Install the nightly version of tensorflow
FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3
WORKDIR /root

RUN pip3 install pandas google-cloud-storage polarTransform pytest

# Copies the trainer code to the docker image.
COPY trainer ./trainer
RUN mkdir trained_models

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-m", "trainer.blackbox_task"]
