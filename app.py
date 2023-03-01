"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OpenAIEmbeddings
from streamlit_extras import add_vertical_space as avs
from langchain.vectorstores import Pinecone
import pinecone
from chain import chat_chain
from config.config_files import APIkeys
from dataclasses import asdict

@st.cache_resource
def init_vectorstore():
    """Initializes vectorstore"""
    print("Initializing vectorstore...")
    pinecone.init(api_key=APIkeys.PineconeAPI, environment=APIkeys.PineconeEnv)
    index = pinecone.Index(APIkeys.PineconeIdx)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_KEY'])
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
    
    return vectorstore


def App(userChoices):
    print("Initializing app...")
    
    # get vectorstore
    vectorstore = init_vectorstore()
    
    # load ChatVectorDBChain
    chain = chat_chain(vectorstore, userChoices.temperature, userChoices.model, userChoices.domain, userChoices.usertitle)

    def get_text():
        input_text = st.text_input("What's on your mind? ", key="input")
        return input_text
        
    user_input = get_text()

    if user_input:
        # run chain with user input and chat history
        output = chain({"question": user_input, 
                        "chat_history": st.session_state["chat_history"],
                        "domain": userChoices.domain,
                        "usertitle": userChoices.usertitle},
                        return_only_outputs=False)
        

        avs.add_vertical_space(5)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output['answer'])

    if st.session_state["generated"]:
        st.session_state["chat_history"].append((user_input, output['answer']))
        st.session_state["sources"]= output['source_documents']
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")