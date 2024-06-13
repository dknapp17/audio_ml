from flask import Flask, request, render_template, redirect, url_for
import torch
import os
import pickle
import AudioUtil as au
import AudioClassifier as ac
import random

app = Flask(__name__)

# Define the uploads folder inside the static directory
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the instrument names dictionary from the pickle file
with open(os.path.join('static', 'instrument_map.pkl'), 'rb') as f:
    instrument_dict = pickle.load(f)

torch.manual_seed(17)
random.seed(17)

# Load the pre-trained model once at the start
trained_model = ac.AudioClassifier()
trained_model.load_state_dict(torch.load('instrmnt_clssfr.pth'))
trained_model.eval()

def extract_features(audio_path):
    aud = au.AudioUtil.open(audio_path)
    reaud = au.AudioUtil.resample(aud, 44100)
    rechan = au.AudioUtil.rechannel(reaud, 2)
    dur_aud = au.AudioUtil.pad_trunc(rechan, 4000)
    shift_aud = au.AudioUtil.time_shift(dur_aud, 0)
    sgram = au.AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    # don't augment prediction sgram
    aug_sgram = au.AudioUtil.spectro_augment(sgram, max_mask_pct=0.0, n_freq_masks=0, n_time_masks=0)
    
    # Normalize the spectrogram
    inputs = aug_sgram.unsqueeze(0)
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    normalized_inputs = (inputs - inputs_m) / inputs_s
    return normalized_inputs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    audio_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(audio_path)
    
    features = extract_features(audio_path)
    
    # Disable gradient updates
    with torch.no_grad():
        # Get predictions
        outputs = trained_model(features)
        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs, 1)
        
    instrument = prediction[0].item()  # Convert tensor to integer

    # Map the predicted class index to the instrument name using the dictionary
    instrument_name = instrument_dict.get(instrument, "Unknown Instrument")
    
    # Calculate the confidence score
    confidence_score = round(torch.softmax(outputs, dim=1)[0][instrument].item() * 100,2)
    
    return render_template('result.html', 
                           instrument=instrument_name,
                           confidence=confidence_score, 
                           file_path=f'uploads/{file.filename}', 
                        #    image_path=image_path
                           )

if __name__ == '__main__':
    app.run(debug=True)
