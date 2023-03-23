"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OpenAIEmbeddings
from streamlit_extras import add_vertical_space as avs
from langchain.vectorstores import Pinecone, Qdrant
import pinecone
from chain import vector_chain, simple_seq_chain
from config.config_files import APIkeys
from dataclasses import asdict
from qdrant_client import QdrantClient
import openai
from datetime import datetime
import asyncio
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from streamingCallback import StreamingCallbackHandler


from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



def build_history(history):
    full_history = ""
    for chat in history:
        full_history += f"Question: {chat[0]} \n Answer: {chat[1]} \n\n"

    print(full_history)
    return full_history


# @st.cache_resource
def init_vectorstore(domain):
    """Initializes vectorstore"""
    print("Initializing vectorstore...")
    
    collections = {
        "Yeti Academy": "sjk-yeti-academy",
        "Typing Agent": "sjk-typing-agent",
    }
    
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_KEY'])

    client = QdrantClient(
        host="0a09d71f-4613-432e-b559-5c8ed3c24617.us-east-1-0.aws.cloud.qdrant.io",
        api_key="oXwue_DbXUWr505z-GnaGkqJTPzy1f34C1SZZk0vTtvHRMwlERJxfA",
    )
    vectorstore = Qdrant(client, collections[domain], embeddings.embed_query,
                        content_payload_key='text', metadata_payload_key='metadata')
    return vectorstore


def init_session_messages():
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y/%m/%d')
    # messages =    [
    #     SystemMessage(content=f"You are a helpful assistant that answers any question about real world scenarios. Keep in mind today is {formatted_date}"),
    # ]
    
    
    messages=[
        {"role": "system", "content": f"You are a helpful assistant that answers any question about real world scenarios. Keep in mind today is {formatted_date}. Be as concise as possible"},
    ]
    
    if 'messages' not in st.session_state:
        st.session_state.messages = messages
    
    return messages

def App(userChoices):
    print("Initializing app...")
    
    messages = init_session_messages()
    
    # load ChatVectorDBChain
    if userChoices.domain != "General":
        # get vectorstore
        vectorstore = init_vectorstore(domain=userChoices.domain)
        vec_chain = vector_chain(vectorstore, userChoices.temperature, domain=userChoices.domain)
    else:
        gen_chain, standalone_creator, streaming_llm_turbo  = simple_seq_chain(userChoices.temperature, domain=userChoices.domain)


    user_input = st.text_input("What's on your mind?", key='widget')

    if st.button("Get response") and user_input != "":
        
        if userChoices.domain != "General":
            # run chain with user input and chat history
            with st.spinner('Generating response...'):
                output = vec_chain({"question": user_input,"chat_history": st.session_state["chat_history"],}, return_only_outputs=False)

            st.session_state.past.append(user_input)
            
            answer = output['answer']
            st.session_state["sources"] = output['source_documents']

        else:
            build_history(st.session_state['chat_history'])

            def get_response(user_input):
                # standalone = standalone_creator({'question': user_input, 'chat_history': st.session_state['chat_history']})
                
                st.session_state['messages'].append({"role": "user", "content": user_input})
                
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state['messages'],
                )
                
                st.session_state['messages'].append({"role": "assistant", "content": response['choices'][0]['message']['content']})
                
                return response['choices'][0]['message']['content']

            with st.spinner('Generating response...'):
                answer = get_response(user_input)
                
            st.session_state["sources"] = []
            st.write(answer)

        st.session_state["chat_history"].append((user_input, answer))
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)

    avs.add_vertical_space(5)
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + "_user")
