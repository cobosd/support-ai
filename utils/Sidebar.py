import streamlit as st
from config.config_files import ModelParams

def sidebar():
    
    
    # INRTO
    html_temp = """<div style="background-color:{};padding:1px"></div>"""
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Ask any question regarding TypingAgent topics and we'll do our best to guide you
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)


    # create some spacing
    for _ in range(8):
        st.text("")

    
    # PARAMETERS TO BE CHOSEN BY USER
    model  = st.radio(
        "Which model do you want to use?",
        ('text-davinci-003', 'text-curie-001', 'text-babbage-001', 'text-ada-001'))

    temperature= st.slider('Temperature', min_value=0.00, max_value=1.00, value=0.00, format='%.2f')
     
    usertitle = st.radio(
        "What\'s your position?",
        ('District administrator', 'School administrator', 'Teacher', 'Student'))
    
    domain = st.radio(
        "What\'s your question about?",
        ('Typing Agent', 'Yeti Academy', 'General'))


    return temperature, usertitle, domain, model