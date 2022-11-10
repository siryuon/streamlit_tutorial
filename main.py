import streamlit as st
from inference import load_model, run
from streamlit_chat import message
from random import random

model, tokenizer = load_model()

st.title('Streamlit STS')
if 'st1' not in st.session_state:
    st.session_state['st1'] = ''
if 'st2' not in st.session_state:
    st.session_state['st2'] = ''
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.form(key='my_form', clear_on_submit=True):
    col1, col2 = st.columns([8, 1])

    with col1:
        st.text_input(
            'Sentence 1',
            key='st1',
        )
        st.text_input(
            'Sentence 2',
            key='st2',
        )

    with col2:
        st.write("&#9660;&#9660;&#9660;")
        submit = st.form_submit_button(label="Ask")

if submit:
    msg1 = (st.session_state['st1'], True)
    msg2 = (st.session_state['st2'], True)
    st.session_state.messages.append(msg1)
    st.session_state.messages.append(msg2)
    for msg in st.session_state.messages:
        message(msg[0], is_user=msg[1], key=random())

    print(msg1[0], msg2[0])
    with st.spinner('연산중'):
        result = run(tokenizer, model, msg1[0], msg2[0])
    print(result)
    msg = (result, False)
    st.session_state.messages.append(msg)
    message(msg[0], is_user=msg[1], key=random())