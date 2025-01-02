import joblib
import streamlit as st
import random
import json

# Load the model and vectorizer
clf = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load the JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)
    

# Define chatbot response function
def chatbot(input_text):
    try:
        # Transform the input text using the vectorizer
        input_text = vectorizer.transform([input_text])
        
        # Predict the intent tag
        tag = clf.predict(input_text)[0]

        # Search for a matching intent and return a random response
        for intent in intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

        # Fallback response if no matching intent is found
        return "I am still learning and couldn't process that. Can you try again?"

    except Exception as e:
        # Return an error fallback response in case of an exception
        return "Oops! Something went wrong. Please try again later."


# Streamlit application
def main():
    # Page configuration
    st.set_page_config(page_title="Intent Based Chatbot 🤖", layout="wide")

    st.title("Intent Based Chatbot 🤖")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Create a container for the chat history
    chat_container = st.container()

    # Add user input box
    with st.form(key="user_input_form"):
        user_input = st.text_input("Chat", placeholder="Type your message here...")
        submit_button = st.form_submit_button("Submit")

    # Process user input
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"sender": "user", "message": user_input})

        # Generate chatbot response
        response = chatbot(user_input)
        st.session_state.chat_history.append({"sender": "bot", "message": response})

    # Display chat history with styled bubbles
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["sender"] == "user":
                st.markdown(
                    f"""
                    <div style='text-align: right; margin: 10px;'>
                        <span style='background-color: #007bff; color: white; padding: 8px 12px; border-radius: 10px;'>
                            {chat['message']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='text-align: left; margin: 10px;'>
                        <span style='background-color: #f44336; color: white; padding: 8px 12px; border-radius: 10px;'>
                            {chat['message']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
