import streamlit as st
from functions.getValues import getValues
import numpy as np

    
def widgets(column):
    
    # further subdivide one of the cells
    # subcol1, subcol2 = column.columns(2)
    
    st.markdown("""
    <style>
        .empty-message {
            font-size: 16px;
            font-weight: 'bold';
            text-decoration: underline;
        }
    </style>""", unsafe_allow_html=True)
    
    if st.session_state['sources']:
        
        with st.expander("Full prompt used"):
            pass
            # st.write(st.session_state.full_prompt[-1])

        
        with st.container():
            for index, item in enumerate(st.session_state['sources']):
                with st.expander(f"{index + 1} — {item.metadata['title']}"):
                    st.write(item.metadata['url'])
                    st.write(item.page_content)
                
    else:
        with st.expander("Prompt used"):
            st.write('Last prompt used will appear here')

        with st.expander("Contexts retrieved"):
            st.write('Contexts will appear here')
                
