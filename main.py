"""Python file to serve as the frontend"""
import streamlit as st
from interface import UI

if __name__ == "__main__":
    st.set_page_config(page_title="Support.ai Demo", page_icon=":robot:", layout='wide')
    st.header("Support.ai Demo")
    
    
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []
        
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        
    if "sources" not in st.session_state:
        st.session_state["sources"] = []
        
    UI()
