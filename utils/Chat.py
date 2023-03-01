import streamlit as st
import openai
from streamlit_chat import message
import json
from utils.GetContext import getContext
from functions.getValues import getValues
from functions.conversation import Convo
from langchain import OpenAI, PromptTemplate, LLMChain

def chat(temperature, usertitle, domain, model, RESPONSE_CONFIG):
    
    #initiate conversation
    convo = Convo(openai)
    
    # selected = pills("", ["Show History", "Don't show History"])
    # Import available prompts
    with open('constants/prompts.json', 'r') as file:
        json_string = file.read()
        prompt_templates = json.loads(json_string)

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'data' not in st.session_state:
        st.session_state['data'] = []
    if 'full_prompt' not in st.session_state:
        st.session_state['full_prompt'] = []
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # # Button styling
    st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #8DF8C2;
            border-color: #8DF8C2;
            color: #272823;
            border-width: 0px;
            border-radius: 20px;
            min-width: 100px;
            font-weight: 'bold';
            font-family: 'Helvetica';
            transition: all 0.5s ease;
        }
        div.stButton > button:hover {
            background-color: #6EBDF7;
            border-color: #6EBDF7;
            color: #272823;
            # color: #fefefe;
            border-radius: 20px;
            border-width: 0px;
            min-width: 130px;
            font-weight: 'bold';
            font-family: 'Helvetica';
            transition: all 0.5s ease;
            }
        div.stButton > button:focus:not(:active){
            background-color: #a7b814;
            border-color: #fefefe;
            border-width: 1px;
            border-radius: 20px;
            min-width: 130px;
            color: #FEFEFE;
            font-weight: 'bold';
            font-family: 'Helvetica';
            transition: all 0.5s ease;
            }
        .response-message {
            font-size: 15px;
            font-style: italic;
        }
    </style>""", unsafe_allow_html=True)
    
    user_input = st.text_input(label="Type here:", placeholder="What's on your mind?", label_visibility='hidden')

    # if user_input:
    if st.button("Get Answer") and user_input:
        res_box = st.empty()
        report = []
        pinecone_contexts = getContext(user_input, RESPONSE_CONFIG['PINECONE_CLIENT'], RESPONSE_CONFIG['EMBED_MODEL'], RESPONSE_CONFIG['OPENAI_API'])
        
        if domain != 'General':   
            if max(getValues(pinecone_contexts, 'score')) > 0.750:         
                contexts = getValues(pinecone_contexts, 'text')
                
                if len(contexts) > 1:
                    context = "\n\n---\n\n".join(contexts)
                    #   context = ". ".join(contexts)
                else:
                    context = contexts[0]
                
                
                # prompt= """The following is a friendly conversation between a human and an AI.  The AI can use previous chat messages in the 'Current conversation' section as a context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\nCurrent conversation: \n{history} \n\nHuman: {question} \nAI: """
                # full_prompt = prompt_templates['main'].format(usertitle=usertitle, domain=domain, context=context, question=user_input)
                prompt = prompt_templates['main']

                for resp in convo.retrieve(user_input, model, prompt, context=context, usertitle=usertitle, domain=domain):
                    report.append(resp.choices[0].text)
                    result = "".join(report).strip()
                    result = result.replace("\n", "")  
                
                    res_box.markdown(f"<div class='response-message'>{result}</div>",  unsafe_allow_html=True)    
                
                # # Looping over the response
                # for resp in openai.Completion.create(
                #                                     model=model,
                #                                     prompt=full_prompt,
                #                                     max_tokens = 220, 
                #                                     temperature = temperature,
                #                                     stream = True):
                    
                #     report.append(resp.choices[0].text)
                #     result = "".join(report).strip()
                #     result = result.replace("\n", "")  
                    
                #     # res_box.markdown(f'*{result}*')     
                #     res_box.markdown(f"<div class='response-message'>{result}</div>",  unsafe_allow_html=True)     
                    
                convo.setLastResponse(result)
                
                output = "".join(report).strip()
                output = output.replace("\n", "")
                
                # store the output as part of conversation history
                st.session_state.past.append(user_input)
                st.session_state.generated.append(output)
                st.session_state.full_prompt.append(prompt.format(context=context, usertitle=usertitle, domain=domain, question = user_input))

                if contexts != []:
                    st.session_state.data.append(pinecone_contexts)
        else:    
            prompt= """The following is a friendly conversation between a human and an AI.  The AI can use previous chat messages in the 'Current conversation' section as a context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\nCurrent conversation: \n{history} \n\nHuman: {question} \nAI: """
                                    
                                    
            for resp in convo.retrieve(user_input, model, prompt):
                report.append(resp.choices[0].text)
                result = "".join(report).strip()
                result = result.replace("\n", "")  
                
                res_box.markdown(f"<div class='response-message'>{result}</div>",  unsafe_allow_html=True)    
                
            convo.setLastResponse(result)
            
            output = "".join(report).strip()
            output = output.replace("\n", "")
            
            
            # store the output as part of conversation history
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
            st.session_state.full_prompt.append(prompt)
            
            if len(st.session_state['chat_history']) == 1:
                st.session_state['chat_history'] = st.session_state['chat_history'] + '\n' + 'Human: ' + user_input + '\n' +  'AI: ' + output               
            else:
                st.session_state['chat_history'] = 'Human: ' + user_input + '\n' +  'AI: ' + output   
                 
                
                    
    st.markdown("----")
                   
    # if selected == "Show History" and len(st.session_state['generated'])!=0: 
    if st.session_state['generated']:
        st.subheader("Chat history")  
        
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i)+'_bot')
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')      
