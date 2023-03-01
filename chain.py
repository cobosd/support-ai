import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from streamingCallback import StreamingCallbackHandler
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.callbacks.base import CallbackManager
from langchain.chains.question_answering import load_qa_chain
from config.config_files import APIkeys, ModelParams
from langchain.prompts.prompt import PromptTemplate

def get_prompt(domain, usertitle):
    suffix = """Assume I am a {position} at {domain}. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    prompt = """
    {context}

    Question: {question}
    Helpful Answer:"""
    
    final_prompt = suffix.format(position=usertitle, domain=domain) + prompt
    
    print(final_prompt)
    return final_prompt



@st.cache_resource
def chat_chain(_vectorstore, temperature,model,domain,usertitle):
    """Contruct a chat chain to query vectorstore and return the answer. This already includes chat memory and streaming callback""" 
    
    print("Loading conversation chain...")
    
    # initialize simple LLM chain
    llm = OpenAI(temperature=0, openai_api_key=APIkeys.OpenAiAPI, model_name=model)
    
    # prepare streamil LLM
    streaming_llm = OpenAI(temperature=temperature, 
                           openai_api_key=APIkeys.OpenAiAPI, 
                           verbose=True, 
                           streaming=True, 
                           callback_manager=CallbackManager([StreamingCallbackHandler()]),
                           max_tokens=400
                           )
    
    # generate standalone question to include chat history and have better accuracy
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
  
    # define how to utilize vectorstore documents retrived
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)
    
    # define full chat chain: 
    # 1. create standalone question based on current ser input
    # 2. load chat history documents
    # 3. combine standalone question, chat history, and vectorstore documents
    chat = ChatVectorDBChain(vectorstore=_vectorstore, combine_docs_chain=doc_chain, question_generator=question_generator, return_source_documents=True)
    
    return chat
