from transformers import pipeline
import librosa
from num2words import num2words

# change to ur model

pipe = pipeline("automatic-speech-recognition", model="C://Users//Joshua Heng//PycharmProjects//noobsbelow//ML//stt//huggingface-version//whisper-medium-finetunev2", device=0)
def test(path, model):
    audio,sr = librosa.load(path,sr=16000)
    text = model(audio)['text']
    try:
        numbers = ''
        for c in text.split():
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

# returns numbers only (used for sentences with more than one number)
def test1(audio_path, model_path):
    model = pipeline("automatic-speech-recognition", model = model_path, device=0)
    audio,sr = librosa.load(audio_path,sr=16000)
    text = model(audio)['text']
    for word in text:
        if word.isdigit():
            return word

# returns numbers only (used for sentences with only one number)
def test2(audio_path, model_path):
    model = pipeline(model=model_path)
    audio, sr = librosa.load(audio_path, sr=16000)
    text = model(audio)['text']
    try:
        numbers = ''
        for c in text.split():
            if c.isdigit():
                numbers += c
        converted_word = num2words(numbers).upper()
    except:
        pass
    return converted_word

print(test1("C://Users//Joshua Heng//Downloads//Train (1)//Train//audio//train_00106.wav", "C://Users//Joshua Heng//PycharmProjects//noobsbelow//ML//stt//huggingface-version//whisper-medium-finetunev2"))
# print(test1('train_00001.wav', "casual/whisper-tiny-finetunev1"))
# print(test2('train_00001.wav', "casual/whisper-tiny-finetunev1"))