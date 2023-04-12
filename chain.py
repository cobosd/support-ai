import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from streamingCallback import StreamingCallbackHandler
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.callbacks.base import CallbackManager
from langchain.chains.question_answering import load_qa_chain
from config.config_files import APIkeys


@st.cache_resource
def vector_chain(_vectorstore, temperature, domain):
    """Contruct a chat chain to query vectorstore and return the answer. This already includes chat memory and streaming callback""" 
    
    print("Loading vector db conversation chain...", domain)
    
    streaming_llm = OpenAI(temperature=temperature, 
                            openai_api_key=APIkeys.OpenAiAPI, 
                            verbose=True, 
                            max_tokens=1000,
                            streaming=True,
                            callback_manager=CallbackManager([StreamingCallbackHandler()]),
                            )

    question_gen_llm = OpenAI(temperature=temperature, 
                            openai_api_key=APIkeys.OpenAiAPI, 
                            verbose=False, 
                            max_tokens=1000,
                            streaming=True,
                            callback_manager=CallbackManager([StreamingCallbackHandler()]),
                            )
    
    question_generator = LLMChain(llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT)

    # define how to utilize vectorstore documents retrived
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)
    
    chat = ChatVectorDBChain(vectorstore=_vectorstore, combine_docs_chain=doc_chain, question_generator=question_generator, top_k_docs_for_context=4, return_source_documents=True)
    
    return chat



# @st.cache_resource
# def simple_seq_chain(temperature, domain):
#     """Initialize is a chain for general purpose questions"""
    
#     print("Loading sequential conversation chain...")
#     print(domain)
    
#     # Firs, get the standalone question
#     standalone_template = """Given the following conversation and a follow up question, rephrase the follow up questions.
#     Chat History:
#     {chat_history}
#     Follow Up Input: {question}
#     Standalone question:"""
    
    
#     query_template = """Provide a straight-forward answer to the question at the end. Use the example below as a format for your answer and provide any additional relevant information to your response. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
#     Example:
#     Q: who is messi?
#     A: Messi is a professional soccer player from Argentina who currently plays for Paris Saint-Germain and is considered one of the greatest players of all time.
    
#     Question: {standalone_question}
#     Answer:"""
    
    
    
#     # initialize simple LLM chain
#     question_generator_llm_turbo = OpenAIChat(
#                 openai_api_key=APIkeys.OpenAiAPI, 
#                 temperature=temperature, 
#                 streaming=True, 
#                 max_tokens=400)
    
#     #Second, buil LLM to get answer based on standalone question
#     streaming_llm_turbo = OpenAIChat(
#             openai_api_key=APIkeys.OpenAiAPI, 
#             temperature=temperature, 
#             max_tokens=400)


    
#     standalone_prompt = PromptTemplate(input_variables=["chat_history", "question"], template=standalone_template)
    
    
    
#     standalone_chain = LLMChain(llm=question_generator_llm_turbo, prompt=standalone_prompt, output_key="standalone_question")

    

#     query_prompt = PromptTemplate(input_variables=["standalone_question"], template=query_template)
#     response_chain = LLMChain(llm=streaming_llm_turbo, prompt=query_prompt, output_key="response")
    
#     overall_chain = SequentialChain(chains=[standalone_chain, response_chain], 
#                                     input_variables=["chat_history", "question"],
#                                     output_variables=["standalone_question", "response"],
#                                     verbose=True)
    
#     return overall_chain, standalone_chain, streaming_llm_turbo
