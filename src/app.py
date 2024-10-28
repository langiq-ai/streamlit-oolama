import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Title of the Streamlit app
st.title('Ollama Interaction App')
logging.info('Streamlit app title set.')

# Sidebar for user input
st.sidebar.title('Input')
logging.info('Sidebar title set.')

# Template for the prompt
template = """Question: {question}

Answer: Let's think step by step."""
logging.debug(f'Prompt template set: {template}')

# Text input for user message
user_input = st.text_input("Enter your message:")
logging.debug('Text input for user message created.')

# Create a ChatPromptTemplate from the template
prompt = ChatPromptTemplate.from_template(template)
logging.debug('ChatPromptTemplate created from template.')

# Initialize the Ollama model
model = Ollama(model="llama3.2")
logging.info('Ollama model initialized with model llama3.2.')

# Check if the send button is clicked
if st.sidebar.button('Send'):
    logging.info('Send button clicked.')
    if user_input:
        logging.debug(f'User input received: {user_input}')

        # Prepare the data for the prompt
        data = {'question': user_input}
        logging.debug(f'Data prepared for prompt: {data}')

        # Create an LLMChain with the prompt and model
        chain = LLMChain(prompt=prompt, llm=model)
        logging.debug('LLMChain created with prompt and model.')

        # Run the chain to get the response
        response = chain.run(**data)
        logging.info('LLMChain run to get response.')

        # Display the response
        st.write("Ollama response:", response)
        logging.debug(f'Ollama response displayed: {response}')
    else:
        st.write("Please enter a message.")
        logging.warning('No user input provided.')