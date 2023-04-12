"""Python file to serve as the frontend"""
import streamlit as st
from utils.Sidebar import sidebar
from utils.Widgets import widgets
from app import App
from config.config_files import ModelParams
    
            
def UI():
    col1, col2  = st.columns([1, 1], gap='large')
    
    with st.sidebar:
        temperature, domain = sidebar()
        userChoices = ModelParams(temperature=temperature, domain=domain)
        
    with col1:
        App(userChoices)

    with col2:
        widgets(col2)
        pass