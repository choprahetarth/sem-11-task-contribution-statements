
import os
from torch.utils.data import Dataset
from torch import argmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

raw_inputs = ["hello","this is ", "a trained machine learning model"]

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
token='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'


tokenizer = AutoTokenizer.from_pretrained("choprahetarth/SemEval-2021-scibert-contributions", use_auth_token=token)

model = AutoModelForSequenceClassification.from_pretrained("choprahetarth/SemEval-2021-scibert-contributions", use_auth_token=token)

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
logits = model(**inputs).logits

for i,j in zip(logits,raw_inputs):
    print(argmax(i)," here ",j )