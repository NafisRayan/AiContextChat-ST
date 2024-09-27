import joblib
import random
import time
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import requests
from io import BytesIO, TextIOWrapper
import PyPDF2
import docx2txt
import csv

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'









######################################################################################################################################










# Function to scrape data
def scrape_data(url):
    # Send HTTP request and parse content
    response = requests.get(url)
    # print(response)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Scraping logic - use BeautifulSoup to find and extract various types of content
    texts = [element.text for element in soup.find_all(['p', 'a', 'img'])]
    links = [element.get('href') for element in soup.find_all('a') if element.get('href')]
    images = [element.get('src') for element in soup.find_all('img') if element.get('src')]

    # Ensure all lists are of the same length by padding the shorter ones with None
    max_length = max(len(texts), len(links), len(images))
    texts += [None] * (max_length - len(texts))
    links += [None] * (max_length - len(links))
    images += [None] * (max_length - len(images))

    # Create a DataFrame using pandas for texts, links, and images
    data = {'Text': texts, 'Links': links, 'Images': images}
    df = pd.DataFrame(data)

    # return the processed data
    return df

# Function to extract text from a PDF file
def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    num_pages = len(pdf_reader.pages)

    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num] 
        text += page.extract_text() 

    return text.replace('\t', ' ').replace('\n', ' ')

# Function to extract text from a TXT file
def extract_text_from_txt(file_bytes):
    text = file_bytes.decode('utf-8')
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file_bytes):
    docx = docx2txt.process(BytesIO(file_bytes))
    return docx.replace('\t', ' ').replace('\n', ' ')

def extract_text_from_csv(file_bytes, encoding='utf-8'):
    # Convert bytes to text using the specified encoding
    file_text = file_bytes.decode(encoding)

    # Use CSV reader to read the content
    csv_reader = csv.reader(TextIOWrapper(BytesIO(file_text.encode(encoding)), encoding=encoding))
    
    # Concatenate all rows and columns into a single text
    text = ""
    for row in csv_reader:
        text += ' '.join(row) + ' '

    return text.replace('\t', ' ').replace('\n', ' ')










######################################################################################################################################










# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}












######################################################################################################################################












# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# GOOGLE API KEY')

    # take google api key as a text input
    GOOGLE_API_KEY = st.text_input('Google API Key')
    
    if GOOGLE_API_KEY == '':
        # st.error("API Key is not set. Please enter your Google API Key.")
        pass
    else:
        genai.configure(api_key=GOOGLE_API_KEY)




    st.write('# Scrap Link')
    
    # take link as a text input
    url_input = st.text_input('Enter the website URL: ', '')
    if url_input:
        if 'https://' not in url_input:
            url_input = 'https://' + url_input
        scraped_data = scrape_data(url_input)
        paragraph = ' '.join(scraped_data['Text'].dropna())
        url_input = paragraph
        # st.write(scraped_data)
        # st.write(paragraph)


    st.write('# Scrap Link')
    
    # st.write("Upload a PDF, TXT, or DOCX file.")
    uploaded_file = st.file_uploader("Upload a PDF, TXT, or DOCX file.")
    print(uploaded_file)

    if uploaded_file:
    # Get the file extension
        file_name, file_extension = os.path.splitext(uploaded_file.name)

        if file_extension:
            # Extract text based on the file extension
            if file_extension == ".pdf":
                uploaded_file = extract_text_from_pdf(uploaded_file.getvalue())
            elif file_extension == ".txt":
                uploaded_file = extract_text_from_txt(uploaded_file.getvalue())
            elif file_extension == ".docx":
                uploaded_file = extract_text_from_docx(uploaded_file.getvalue())
            elif file_extension == ".csv":
                uploaded_file = extract_text_from_csv(uploaded_file.getvalue())

            else:
                st.error("Unsupported file type.")


    st.write('# Chat History')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    
    
    st.session_state.chat_title = f'ChatSession-{random.randint(1,10)} at {int(time.time())}'













######################################################################################################################################













st.write('# Chat with Gemini')

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []
    print('new_cache made')
st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)













######################################################################################################################################














# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Your message here...'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    











######################################################################################################################################












    if uploaded_file:
        prompt = uploaded_file + ' ' +"Take the given data above, as information and generate a response based on this prompt: " + prompt

    if url_input:
        prompt = url_input + ' ' +"Take the given data above, as information and generate a response based on this prompt: " + prompt  

    if  uploaded_file and url_input:
        prompt = uploaded_file + ' ' + url_input + ' ' +"Take the given data above , as information and generate a response based on this prompt: " + prompt











######################################################################################################################################











    ## Send message to AI
    response = st.session_state.chat.send_message(
        prompt,
        stream=True,
    )
    
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        
        for chunk in response:
            # Simulate stream of chunk
            # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
            if hasattr(chunk, 'text'):
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)
                    # Rewrites with a cursor at end
                    message_placeholder.write(full_response + '▌')
            elif hasattr(chunk, 'parts') and len(chunk.parts) > 0:
                full_response += chunk.parts[0].text
                time.sleep(0.05)
                message_placeholder.write(full_response + '▌')
            else:
                # If no text or parts, wait for the next chunk
                time.sleep(0.1)
        
        # Write full message with placeholder
        message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    












######################################################################################################################################











    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )