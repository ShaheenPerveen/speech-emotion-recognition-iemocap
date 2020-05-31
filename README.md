
# speech-emotion-recognition-iemocap

## Introduction

In this project, I have tried to recognize emotions from audio signals of IEMOCAP dataset. The audio files are in english language. IEMOCAP is an acted, multimodal and multispeaker database, recently collected at SAIL lab at USC. It contains 12 hours of audiovisual information. Different windows of audio sessions has been tagged with one of the eight emotions(frustrated, angry, happiness, sadness, excited, surprised, disgusted and neutral). Details of dataset can be found [here](https://sail.usc.edu/iemocap/). In the approach, I have only utilized the audio part of the data

## Requirements
1. Pytorch
2. torch
3. matplotlib
4. numpy
5. pandas
6. librosa
7. multiprocessing
8. joblib
9. re
10. soundfile

## Details of approach

The objective of the attempt was to evaluate multi-modal implementation using features extracted from audio signals.
Following are the steps applied to arrive at the final model:
- Analyzed audio signals, extracted labels tagged to sessions. Each session has been tagged with different emotions based on timestamp
- Low Level Descriptors are expected to capture emotion related information. Extracted acoustic features like MFCC, Mel, pitch, contrast, flatness, zero crossing rate, chroma and harmonic means. Apart from using these acoustic features directly, I have applied aggregators and summarizers like mean, min, max, standard deviation etc
- These extracted features will be used as input in different models
- Using the audio signal, generated images of mel-spectrogram. A spectrogram is a representation of speech over time and frequency. 2D convolution filters help capture 2D feature maps in any given input. Such rich features cannot be extracted and applied when speech is converted to text and or phonemes. Spectrograms, which contain extra information not available in just text, gives us further capabilities in our attempts to improve emotion recognition
Below is an example:
(/Users/shaheen.perveen/Desktop/Screenshot 2020-05-31 at 7.33.05 PM.png)


