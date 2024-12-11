import streamlit as st

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

import torchaudio

import AudioUtils
import pandas as pd

import librosa
import matplotlib.pyplot as plt
# need to move this to a .py file and import

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=12)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

# Create the model and put it on the GPU if available
myModel = AudioClassifier()

myModel = AudioClassifier()
myModel.load_state_dict(torch.load('myModel.pth', weights_only=True))

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

def predict_single_image(model, image, device):
    """
    Predicts the class of a single image using a trained CNN model.
    
    Args:
        model (torch.nn.Module): The trained model.
        image (torch.Tensor): The input image tensor (C, H, W).
        device (torch.device): The device (CPU or GPU) to perform inference.
        mean (float or torch.Tensor): Mean for normalization (optional).
        std (float or torch.Tensor): Std for normalization (optional).
    
    Returns:
        pred_class (int): The predicted class index.
        pred_probs (list): List of probabilities for each class.
    """
    model.eval()  # Set model to evaluation mode

    # Ensure the input is a batch of size 1 (1, C, H, W)
    if len(image.shape) == 3:  # If input is (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension

    # Move image to device
    image = image.to(device)


    inputs_m, inputs_s = image.mean(), image.std()

    image = (image - inputs_m) / inputs_s

    with torch.no_grad():
        # Get predictions
        outputs = model(image)
        pred_probs = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(pred_probs, dim=1)

    # Convert probabilities to a list and return the predicted class
    return pred_class.item()


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    



music_df = pd.read_csv('./resources/instrument_map.csv')

if uploaded_file is not None:
    # preprocess uploaded file
    st.write('preprocessing')
    aud = AudioUtils.AudioUtil.open(uploaded_file)
    aud_rechannel = AudioUtils.AudioUtil.rechannel(aud,2)
    aud_sg = AudioUtils.AudioUtil.spectro_gram(aud_rechannel, n_mels=64, n_fft=1024, hop_len=None)
    # Predict instrument
    st.write('predicting')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.header("Here's how we visualize your sound:")
    st.write("We'll plot the spectrogram here")
    fig, axs = plt.subplots(2, 1)
    plot_waveform(aud[0], aud[1], title="Original waveform", ax=axs[0])
    plot_spectrogram(aud_sg[0], title="augmented spectrogram", ax=axs[1])
    fig.tight_layout()
    st.pyplot(fig)
    st.header("Here's how the computer sees your sound!")
    pred = predict_single_image(model = myModel, image=aud_sg,device = device)
    st.write(aud_sg)
    pred_string = music_df.loc[music_df['target_instrument']==pred]['instrument_name'].iloc[0]
    st.header(f'The Model Predicts: {pred_string} !')
    img_path = f"./resources/icons/{pred_string}.png"
    st.image(img_path, width=400)
    # outputs = myModel(uploaded_file)


# if uploaded_file is not None:
#     file_ext = uploaded_file.name.split(".")[-1]
#     if file_ext in ["wav"]:
#         with open("temp." + file_ext, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.success("File uploaded successfully")