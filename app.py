
import warnings
warnings.filterwarnings('ignore')

from key import cohere_api_key
from langchain_community.chat_models import ChatCohere

class CustomChatCohere(ChatCohere):
    def _get_generation_info(self, response):
        # Custom handling of generation info
        generation_info = {}
        if hasattr(response, 'token_count'):
            generation_info["token_count"] = response.token_count
        # Add other attributes if needed
        return generation_info

llm = CustomChatCohere(cohere_api_key=cohere_api_key)

from key import serp_api_key
import os
os.environ['SERPAPI_API_KEY'] = serp_api_key
from langchain.agents import AgentType, initialize_agent, load_tools

tools = load_tools(['serpapi','llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

import streamlit as st

st.title('Ask Google')
ask = st.text_area('Ask me anything')

if ask:
    with st.spinner('Searching...'):
        try:
            response = agent.invoke(ask)
            st.markdown(response['output'])
        except ValueError as e:
            st.error(f"An error occurred: {e}")
