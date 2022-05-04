# ImageCaptioning 

This repo contains an implementation of a ImageCaptioning model. It was implemented as a part of 4 ECTS course `Deep Learning` of the [Data Science Bachelor](https://www.fhnw.ch/de/studium/technik/data-science) at the [FHNW](www.fhnw.ch).

## Architecture

The architecture is basically as follows:
- A pretrained CNN-model (e.g. `ResNet50`) is used to generate features from the images.
- With the help of an embedding, the dimension is adapted to the vocab size and the embedding dimension is selected based on available computing resources. Technically, a higher dimension should be better but it takes longer to train and requires more resources.
- This vector is then passed as the first `hidden state` in a LSTM.

## Futher details
Please have a look at [main.ipynb](https://github.com/kenfus/ImageCaptioning/blob/master/main.ipynb)