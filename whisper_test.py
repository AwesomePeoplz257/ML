from transformers import pipeline
from num2words import num2words
import os
import glob
import csv
import torch
import tqdm
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

# try running without this code first. if it does not work, run this code.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# change to ur model
pipe = pipeline(model="AwesomePeoplz257/whisper-medium-finetunev2")

class AudioPreproc(torch.utils.data.Dataset):
    def __init__(self, audio_dir):
        self.dir = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="English", task="transcribe")

    def __getitem__(self, index):
        audio, sr = librosa.load(self.dir[index], sr=16000)
        audio = self.processor(audio, sampling_rate=16000, return_tensors="np").input_features
        # audio = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        # audio = self.processor.decode(audio)
        # print(audio)
        # audio = torch.tensor(audio)
        # audio = np.array(audio)
        return audio, self.dir[index]

    def __len__(self):
        return len(self.dir)

def test(model, audio):
    text = model(audio)['text']
    try:
        numbers = ''
        for c in text:
            if c.isdigit():
                numbers += c
        word = num2words(numbers)
        index = text.index(numbers)
        text = text[:index] + word + text[index + len(numbers):]
    except:
        pass
    text = text.upper()
    text = text[:-1]
    return text


processor = WhisperProcessor.from_pretrained("AwesomePeoplz257/whisper-medium-finetunev2", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("AwesomePeoplz257/whisper-medium-finetunev2")
model.config.forced_decoder_ids = None
def test1(processor, model, audio):
    # input_features = processor(audio, sampling_rate=16000,return_tensors="pt").input_features

    audio = torch.reshape(audio, (16, 80, 3000))
    print(audio[1])
    predicted_ids = model.generate(audio)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    try:
        numbers = ''
        for c in text:
            if c.isdigit():
                numbers += c
        word = num2words(numbers)
        index = text.index(numbers)
        text = text[:index] + word + text[index + len(numbers):]
    except:
        pass
    text = text.upper()
    text = text[:-1]
    return text

# specify the folder containing your audio files
folder_path = "C:\\Users\\Joshua Heng\\Downloads\\Test_Novice\\Test_Novice\\audio"

# specify the output CSV file
output_csv = "C:\\Users\\Joshua Heng\\Downloads\\Test_Novice_19May.csv"

AudioLoader = torch.utils.data.DataLoader(AudioPreproc(folder_path), 16, pin_memory=True)

# open the CSV file in write mode
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # write the header
    writer.writerow(["Audio File", "Transcription"])

    # process each audio file
    for audio, path in tqdm.tqdm(AudioLoader):
        # print(type(audio))
        # audio = audio.cuda()
        # audio = audio.cpu()
        # audio = np.array(audio)
        # print(audio)
        with torch.no_grad():
            transcription = test1(processor, model, audio)
        # write the audio file name and its transcription to the CSV file
        for text in transcription:
            writer.writerow([os.path.basename(path), text])
