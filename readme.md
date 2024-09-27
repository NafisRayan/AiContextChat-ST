# AiContextChat-ST

## Overview

AiContextChat-ST is a chat application built using Streamlit and various natural language processing tools. It provides a platform for users to engage in conversations while leveraging AI capabilities for enhanced interaction.

## Features

- Real-time chat functionality
- Support for uploading PDF, TXT, and DOCX files
- Web scraping capability to gather information from websites
- Integration with Google's Generative AI model for intelligent responses
- Ability to save and retrieve past chats

## Technologies Used

- Streamlit for building the web interface
- BeautifulSoup for HTML parsing and web scraping
- PyPDF2 for PDF text extraction
- docx2txt for DOCX text extraction
- google.generativeai for AI-powered responses
- joblib for saving/loading chat history

## Setup Instructions

1. Clone the repository:
git clone https://github.com/NafisRayan/AiContextChat-ST.git


2. Install dependencies:
pip install streamlit pandas beautifulsoup4 google-generativeai py-pdf2 docx2txt joblib


3. Run the application:
streamlit run app.py


4. Open the application in your web browser at `http://localhost:8501`.

## Usage

1. Enter a URL or upload a file to gather information.
2. Select a past chat or start a new one.
3. Type your message in the chat input area.
4. The AI assistant will respond based on the gathered information and your query.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or issues.