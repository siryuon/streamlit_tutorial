import sys

sys.path.append('model')
from typing import Callable, Dict, List, NoReturn, Tuple
import numpy as np
import streamlit as st
from datasets import Dataset, DatasetDict, Features, Value, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

@st.cache(allow_output_mutation=True)
def load_model():
    model_name = 'snunlp/KR-ELECTRA-discriminator'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, max_length=100)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model, tokenizer

def run(tokenizer, model, st1, st2):
    resultLst = ['비슷한 문장이 아닌 것 같은데?', '비슷한 문장인 것 같아!']
    text = st1 + '[SEP]' + st2
    token = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=100,
                                      truncation=True)
    pred = model(token['input_ids'])[0]
    print(pred)
    result = pred.argmax().item()

    return resultLst[result]