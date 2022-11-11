import sys

sys.path.append('model')

import streamlit as st
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from model import Model

@st.cache(allow_output_mutation=True)
def load_model():
    model_name = 'snunlp/KR-ELECTRA-discriminator'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, max_length=100)
    model = Model.load_from_checkpoint(checkpoint_path='model.ckpt')
    return model, tokenizer

def run(tokenizer, model, st1, st2):
    resultLst = ['비슷한 문장이 아닌 것 같은데?', '비슷한 문장인 것 같아!']
    text = st1 + '[SEP]' + st2
    token = tokenizer(text, add_special_tokens=True, return_tensors='pt', max_length=100, truncation=True)['input_ids']
    pred = model(token)

    label = abs(round(float(pred), 1))
    if label < 0 :
        label = 0.0
    elif label >= 5:
        label = 5.0

    return label
