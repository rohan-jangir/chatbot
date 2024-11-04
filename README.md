# Ganesh Chatbot

## Overview
Ganesh is a simple and casual conversation chatbot built using Python. It utilizes Natural Language Processing (NLP) techniques to understand user input and respond appropriately. The chatbot is designed to assist users in various topics through predefined intents and patterns.

## Features
- **Greeting and Farewell Responses**: The chatbot can initiate conversations and respond to farewells.
- **Assistance Queries**: Users can ask for help or information, and the chatbot will provide relevant responses.
- **Emotion Acknowledgment**: The bot can recognize and respond to users expressing different emotions.
- **Recommendations**: It can provide suggestions for movies, restaurants, and more.
- **Interactive User Interface**: Built with Streamlit for an easy-to-use chat interface.

## Technologies Used
- Python
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Streamlit
- Logistic Regression
- TfidfVectorizer

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rohan-jangir/chatbot/tree/main
   ```

**2. Install required packages:**
```bash
pip install -r requirements.txt
```
**3. Download NLTK Data:** Run the following command in a Python shell or script to download the necessary NLTK data:
```bash
import nltk
nltk.download('punkt')
```
## Usage
To run the chatbot, execute the following command:
```bash
streamlit run app.py
```
This will start a local server, and you can interact with the chatbot through your web browser.

## How It Works
**1. Intent Definition:** The chatbot's behavior is defined through a list of intents, each containing patterns that trigger responses.

**2. Model Training:** The input patterns are vectorized using TfidfVectorizer, and a Logistic Regression model is trained on the processed data.

**3. User Interaction:** The user inputs messages, which are transformed and fed into the trained model to predict the intent. A response is then randomly selected from the corresponding intent's responses.

## Acknowledgments
* Streamlit for providing an easy way to create web apps.
* NLTK for Natural Language Processing tools.
* Scikit-learn for machine learning functionality.
