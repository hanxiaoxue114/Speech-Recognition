# Speech Recognition

## Overview

This repository includes the implementation of various speech recognition models, such as an RNN, DNN, bidirectional RNN, and a hybrid CNN-RNN.

## Datasets

- [**LibriSpeech**:](https://www.openslr.org/12) A corpus of approximately 1000 hours of 16kHz read English speech and corresponding text data. 
The raw audios are converted into mel frequency cepstral coefficients (MFCCs) using the python\_speech\_features library and into spectral signals with NumPy. 


## Usage

Instructions for training and evaluating the models:

```bash
python main.py
