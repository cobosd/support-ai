import streamlit as st
from langchain import PromptTemplate
from time import time

## This will be implemented in the main code soon

@st.cache_resource
class Convo:
    """conversation memory."""
    
    def __init__(self, LLM):
        self.llm = LLM
        self.last_generator = None 
        self.last_input = None
        self.last_response = ''
        self.history = ''
        self.prompt = ''
        # self.prompt = PromptTemplate(template=template, input_variables=["history","question"])
        
        
    def __updateHistory(self):
        if self.last_input != None:
            self.history =  self.history + '\n' + f"Human: {self.last_input}" + '\n' + f"AI: {self.last_response}"
        
        
    def setLastResponse(self, response):
        self.last_response = response
        self.__updateHistory()
    
    def retrieve(self, user_input, model, prompt, context=None, usertitle=None, domain=None):
        self.last_input = user_input
        self.model = model
        self.prompt = prompt
        
        if not context:
            print('no context')
            self.last_generator = self.llm.Completion.create(
                        model=self.model,
                        prompt=self.prompt.format(history= self.history, question=str(user_input).strip()),
                        max_tokens = 220, 
                        temperature = 0,
                        stream = True)
        else:
            print('context')
            self.last_generator = self.llm.Completion.create(
                        model=self.model,
                        prompt=self.prompt.format(question=str(user_input).strip(), context=context, usertitle=usertitle, domain=domain),
                        max_tokens = 220, 
                        temperature = 0,
                        stream = True)
            
        return self.last_generator
        