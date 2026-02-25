import streamlit as st
from components.upload import render_uploader
from components.history import render_history_download
from components.chatui import render_chat


st.set_page_config(page_title="AI Medical Assistant",layout="wide")
st.title(" 🩺 Medical Assistant Chatbot")


render_uploader()
render_chat()
render_history_download()