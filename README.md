# About this project
I built a classifier using a convolutional neural network (CNN) to identify an instrument playing in an audio clip. I sourced over 7,000 labeled audio clips (6.7GB of data) using the freesound API from freesound.org. Using pytorch, I created a CNN that can identify audio clips as one of 12 instruments (cello, clarinet, double bass, flute, oboe, piccolo, alto sax, baritone, sax, soprano sax, tenor sax, trumpet, violin). The end result is a streamlit app that allows a user to upload an audio file, and be returned a prediction for the instrument. The app can be accessed [here][1]. While the app works on small audio files for a limited subset of instruments, this is a framework that can be extended to more instruments with more data. I consider this project step 1 in a larger project to do full music transcription of multiple instruments. As an amateur guitar player, I'm extremely interested in the intersection of music and math!

# About the data
The data is sourced from freesound.org, particularly from the user MTG. MTG refers to The Music Technology Group, affiliated with the Universitat Pompeu Fabra in Barcelona. This group provided the audio files for research into audio processing. I randomly assigned 80% of the data to put in the training set and 20% to be in the validation set.

# Approach
The secret to classifying audio is to treat it as a computer vision problem. Audio can be represented as what's called a Mel Spectrogram. I'm not going to get into all the technical aspects here, but a Mel Spectrogram essentially takes an audio clip and applies transformations to represent it as a heatmap. Here's an example:

![alt text](https://github.com/dknapp17/audio_ml/blob/main/resources/readme/spectrogram.png?raw=true)


Notice that the spectrogram is a plot of frequency over time, having a color that represents an amplitude. Given a lot of Mel Spectrograms, the CNN can learn the features that identify different instruments. For instance, violins tend to have different frequency/amplitude combinations than cellos.

## Design choices.
I chose to use a CNN to get experience building a pytorch neural network from scratch. Other options would have been to fine tune a pretrained neural network or use a transformer architecture and may be explored in the future. 

## CNN Architecture 

Here is what the CNN architecture looks like:

And here is what is happening at each step 

Convolution:
Pooling:
Linear:

In image recognition, it is common to go from large 

# Reproducibility
This project is designed to be reproducible. That is, you can follow the instructions to download the data in the same way that I did and train a model from scratch. See the "Getting Started" section to learn more.


# Getting Started:

## Create a Freesound Account

Go to https://freesound.org and click "Join"

## Request a freesound API key

Go to https://freesound.org/help/developers/ and select "Get API Key". After filling out a form, you should receive a Client id and a Client Secret/API Key. These 2 values will be needed later

## Fork the repo
Fork the repo into your local machine

## Set up local environment

1) Create the freesound folder 
This folder will be used to store audio files on your local machine

Open a terminal window, navigate to your repo and run the following 
~~~
mkdir ./freesound
~~~

2)  Create python virtual environment
~~~
python -m venv aml_venv
~~~

3)   Activate the environment
~~~
./aml_venv/Scripts/activate
~~~

4) Install Packages
~~~
pip install -r requirements.txt
~~~

5) Modify your fs_config.py file in the config folder.
   First, run the following:
   ~~~
   git update-index --skip-worktree config/fs_config.py
   ~~~
   This ensures that when you update the config file, the passwords don't get tracked and uploaded to github on commits.
   Change the fs_cid variable to your client_id you got when you requested your freesound API key
   Change the fs_client_secret variable to your client secret you got when you requested your freesound API key

You're ready to begin!


# Building and training the model
The audio_ml_nn.ipynb notebook demonstrates how to build a neural network for audio classification. In the future this will be extended to other applications


[1]: <https://instrumentclassify.streamlit.app/>

