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


def build_history(history):
    full_history = ""
    for chat in history:
        full_history += f"Question: {chat[0]} \n Answer: {chat[1]} \n\n"

    print(full_history)
    return full_history


@st.cache_resource
def init_vectorstore():
    """Initializes vectorstore"""
    print("Initializing vectorstore...")
    # pinecone.init(api_key=APIkeys.PineconeAPI, environment=APIkeys.PineconeEnv)
    # index = pinecone.Index(APIkeys.PineconeIdx)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['OPENAI_KEY'])
    # vectorstore = Pinecone(index, embeddings.embed_query, "text")

    client = QdrantClient(
        host="0a09d71f-4613-432e-b559-5c8ed3c24617.us-east-1-0.aws.cloud.qdrant.io",
        api_key="oXwue_DbXUWr505z-GnaGkqJTPzy1f34C1SZZk0vTtvHRMwlERJxfA",
    )
    COLLECTION = "sjk-typing-agent"

    vectorstore = Qdrant(client, COLLECTION, embeddings.embed_query,
                        content_payload_key='text', metadata_payload_key='metadata')
    return vectorstore


def App(userChoices):
    print("Initializing app...")

    # get vectorstore
    vectorstore = init_vectorstore()

    # load ChatVectorDBChain
    if userChoices.domain != "General":
        vec_chain = vector_chain(vectorstore, userChoices.temperature)
    else:
        gen_chain, sq  = simple_seq_chain(userChoices.temperature)

    def get_text():
        input_text = st.text_input("What's on your mind? ", key="input")
        return input_text

    user_input = get_text()
    answer = ""

    if user_input:
        if userChoices.domain != "General":
            # run chain with user input and chat history
            output = vec_chain({"question": user_input,
                            "chat_history": st.session_state["chat_history"],
                            # "domain": userChoices.domain,
                            # "usertitle": userChoices.usertitle
                            },
                            return_only_outputs=False)

            st.session_state.past.append(user_input)
            answer = output['answer']
            st.session_state["sources"] = output['source_documents']

        else:
            print(sq({'question': user_input, 'chat_history': st.session_state['chat_history']}))
            history_as_string = build_history(st.session_state['chat_history'])
            output = gen_chain(
                {"question": user_input, "chat_history":  history_as_string})

            answer = output["response"]
            st.session_state["sources"] = []
            st.write(answer)

        st.session_state["chat_history"].append((user_input, answer))
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)

    avs.add_vertical_space(5)
    if st.session_state["generated"]:
        print('here')
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + "_user")
