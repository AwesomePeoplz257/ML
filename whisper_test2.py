from transformers import pipeline
import librosa
from num2words import num2words
import os
import glob
import csv

# try running without this code first. if it does not work, run this code.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# change to ur model
pipe = pipeline(model="openai/whisper-large-v2", device=0) # to run on GPU

def test(path, model):
    audio,sr = librosa.load(path,sr=16000)
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

# specify the folder containing your audio files
folder_path = "C:\\Users\\Joshua Heng\\Downloads\\Test_Novice\\Test_Novice\\audio\\"

# specify the output CSV file
output_csv = "C:\\Users\\Joshua Heng\\Downloads\\Test_Novice_19May.csv"

# get a list of all audio files in the folder
audio_files = glob.glob(os.path.join(folder_path, "*.wav"))

# open the CSV file in write mode
with open(output_csv, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # write the header
    writer.writerow(["Audio File", "Transcription"])
    index = 0
    # process each audio file
    for audio_file in audio_files:
        if index < 2405:
            index+=1
            continue
        index+=1
        transcription = test(audio_file, pipe)
        # write the audio file name and its transcription to the CSV file
        print(index, transcription)
        writer.writerow([os.path.basename(audio_file), transcription])