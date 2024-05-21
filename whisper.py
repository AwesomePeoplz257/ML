from model_loader import *
import torch
from transformers import pipeline, WhisperFeatureExtractor, WhisperTokenizer
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor


def preprocess_data(batch):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor([instance['array'] for instance in audio], sampling_rate=16000).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

dataset = load_dataset("casual/national_speech_corpusv2")
dataset = dataset['train']
trainset = Dataset.from_dict(dataset[:int(3538 * 0.8)])
testset = Dataset.from_dict(dataset[int(3538 * 0.8):])
trainset = preprocess_data(trainset)

print(trainset)