import os
import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Download NLTK data (run only once)
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey"],
        "responses": ["Hi there, How can i assist you", "Hello, How can i help you", "Hey,  How can i help you"]
    },
    {
        "tag": "greeting2",
        "patterns": ["How are you", "What's up"],
        "responses": ["I'm fine, thank you, How can i assist you", "Nothing much"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "What are you", "What is your purpose"],
        "responses": ["My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "ask2",
        "patterns": ["Who are you", "What is your name"],
        "responses": ["I am Ganesh, a chatbot", "I am Ganesh, How can i assist you"]
    },

    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey there"],
        "responses": ["Hello! How can I help you today?"]
    },
    {
        "tag": "asking_for_help",
        "patterns": ["Can you help me?", "I need assistance"],
        "responses": ["Of course! What do you need help with?"]
    },
    {
        "tag": "gratitude",
        "patterns": ["Thank you", "Thanks a lot", "Appreciate it"],
        "responses": ["You're welcome! Feel free to ask if you need anything else."]
    },
    {
        "tag": "introduction",
        "patterns": ["My name is [Name]", "I'm [Name]", "Call me [Name]"],
        "responses": ["Nice to meet you,! How can I assist you today?"]
    },
    {
        "tag": "asking_for_information",
        "patterns": ["What's the weather like?", "Can you tell me about [topic]?"],
        "responses": ["Provide relevant information based on the query."]
    },
    {
        "tag": "expressing_emotions",
        "patterns": ["I'm happy", "I'm sad", "I'm frustrated"],
        "responses": ["Acknowledge the emotion and offer support or encouragement."]
    },
    {
        "tag": "jokes",
        "patterns": ["Tell me a joke", "Do you know any funny stories?"],
        "responses": ["Share a light-hearted joke or amusing anecdote."]
    },
    {
        "tag": "asking_for_recommendations",
        "patterns": ["What movie should I watch?", "Can you suggest a good restaurant?"],
        "responses": ["Provide recommendations based on preferences or criteria."]
    },
    {
        "tag": "acknowledging_responses",
        "patterns": ["Okay", "Got it", "Thanks for letting me know"],
        "responses": ["Confirm understanding and continue the conversation as needed."]
    },
    {
        "tag": "asking_for_name",
        "patterns": ["What is your name", "May I know your name"],
        "responses": ["You can call me ram."]
    },
    {
        "tag": "asking_for_age",
        "patterns": ["How old are you?", "What is your age?"],
        "responses": ["I don't have an age. I'm just here to assist you."]
    },
    {
        "tag": "asking_for_location",
        "patterns": ["Where are you located?", "What is your location?","Where are you from","Where you living"],
        "responses": ["I exist in the digital realm, so I don't have a physical location."]
    },
    {
        "tag": "asking_for_time",
        "patterns": ["What time is it?", "Can you tell me the time?"],
        "responses": ["Sorry, I don't have access to real-time information like the current time."]
    },
    {
        "tag": "expressing_gratitude",
        "patterns": ["That's helpful", "I appreciate your assistance"],
        "responses": ["Glad I could assist you!"]
    },
    {
        "tag": "asking_for_contact",
        "patterns": ["Can I contact you?", "Do you have a contact email?"],
        "responses": ["I'm just a bot and don't have a contact email. Is there something else I can help you with?"]
    },
    {
        "tag": "asking_for_feedback",
        "patterns": ["Do you want feedback?", "Can I give you feedback?"],
        "responses": ["Feedback is always welcome! Feel free to share your thoughts."]
    },
    {
        "tag": "asking_for_weather",
        "patterns": ["What's the weather forecast?", "How's the weather today?"],
        "responses": ["I'm sorry, I don't have access to real-time weather data."]
    }
]


    # Add more intents...

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.title("Ganesh")
    st.write('''Hi,
             Welcome to the ganesh. A simple and casual conversation chatbot.
             Please type a message and press Enter to start the conversation.''')

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()
if __name__ == '__main__':
    main()


