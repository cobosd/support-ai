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
from langchain.chains import SequentialChain


@st.cache_resource
def vector_chain(_vectorstore, temperature,model,domain,usertitle):
    """Contruct a chat chain to query vectorstore and return the answer. This already includes chat memory and streaming callback""" 
    
    print("Loading vector db conversation chain...")
    
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

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    
    # define how to utilize vectorstore documents retrived
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)
    
    # define full chat chain: 
    # 1. create standalone question based on current ser input
    # 2. load chat history documents
    # 3. combine standalone question, chat history, and vectorstore documents
    chat = ChatVectorDBChain(vectorstore=_vectorstore, combine_docs_chain=doc_chain, question_generator=question_generator, return_source_documents=True)
    
    return chat




@st.cache_resource
def simple_seq_chain(temperature, model):
    print("Loading sequential conversation chain...")
    
    # initialize simple LLM chain
    question_generator_llm = OpenAI(model_name=model, 
                openai_api_key=APIkeys.OpenAiAPI, 
                temperature=temperature, 
                streaming=True, 
                callback_manager=CallbackManager([StreamingCallbackHandler()]),
                max_tokens=400)
    
    # Firs, get the standalone question
    standalone_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    
    standalone_prompt = PromptTemplate(input_variables=["chat_history", "question"], template=standalone_template)
    print(standalone_prompt)
    standalone_chain = LLMChain(llm=question_generator_llm, prompt=standalone_prompt, output_key="standalone_question")
    
    
    #Second, buil LLM to get answer based on standalone question
    streaming_llm = OpenAI(model_name=model, 
            openai_api_key=APIkeys.OpenAiAPI, 
            temperature=temperature, 
            streaming=True, 
            callback_manager=CallbackManager([StreamingCallbackHandler()]),
            max_tokens=400)
    
    query_template = """Respond to the folling question in detail. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Question: {standalone_question}
    Helpful Answer:"""
    
    query_prompt = PromptTemplate(input_variables=["standalone_question"], template=query_template)
    response_chain = LLMChain(llm=streaming_llm, prompt=query_prompt, output_key="response")
  
        
    overall_chain = SequentialChain(chains=[standalone_chain, response_chain], 
                                    input_variables=["chat_history", "question"],
                                     # Here we return multiple variables
                                    output_variables=["standalone_question", "response"],
                                    verbose=True)
    
    return overall_chain
