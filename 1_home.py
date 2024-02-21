import os
import streamlit as st
import openai

from core.pipeline_builder import build_query_pipeline
from core.index_builder.inquiry_index_builder import load_inquiry_index
from core.index_builder.act_index_builder import load_act_index
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

import llama_index.core

os.environ["OPENAI_API_KEY"] = st.secrets.openai_key

llama_index.core.set_global_handler("arize_phoenix")

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
)
Settings.embed_model = embed_model

st.set_page_config(
    page_title="ICT Construction Chatbot",
    page_icon="👷",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
openai.api_key = st.secrets.openai_key
st.title("ICT 건설 컨설턴트, powered by Wordbricks 👷💬")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "궁금한 사항을 물어보세요. ICT 건설에 대한 전문가봇이 답변해드립니다.",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_indexes():
    with st.spinner(text="데이터를 로딩중 입니다. 잠시만 기다려주세요."):
        inquiry_index = load_inquiry_index()
        act_index = load_act_index()
        return {"inquiry": inquiry_index, "act": act_index}


indexes = load_indexes()
qp = build_query_pipeline(indexes)

if "query_pipeline" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.query_pipeline = qp

if prompt := st.chat_input("질문"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("생각중..."):
            response = st.session_state.query_pipeline.run(query_str=prompt)
            st.write(str(response))
            message = {"role": "assistant", "content": str(response)}
            st.session_state.messages.append(message)  # Add response to message history
