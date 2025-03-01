try:
    import streamlit as st
except ImportError:
    st.error("❌ Error: Streamlit library is not installed. Please install it using 'pip install streamlit'.")
import requests

# Streamlit UI
st.title("🦠 COVID Chatbot")
st.write("Ask about COVID-19 variants and outbreak predictions.")

# User input
user_input = st.text_input("Ask a question:", "")

if st.button("Send"):
    if user_input:
        # Send request to Flask API
        api_url = "http://127.0.0.1:5000/chatbot"
        response = requests.post(api_url, json={"message": user_input})
        
        # Display chatbot response
        if response.status_code == 200:
            chatbot_reply = response.json().get("response", "No response")
            st.success(f"🤖 Chatbot: {chatbot_reply}")
        else:
            st.error("❌ Error: Unable to connect to the chatbot API.")