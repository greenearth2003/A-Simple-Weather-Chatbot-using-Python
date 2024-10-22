import streamlit as st
import model 
import os

predef_emb_path = "./predefined_input_emb.npy"
predef_input_path = "./predefined_input.txt"
predef_response_path = "./predefined_response.txt"

# Create predefined input embedding files
model.get_predefined_input_embeddings(predef_input_path, predef_emb_path)

# Read predef_response file: contains predefined respones for the question about the weather
predef_responses = []
with open(predef_response_path, 'r', encoding="utf-8") as file:
    for line in file:
        predef_responses.append(line)

def main():
    st.title("Simple BERT-powered Weather-Chatbot")
    st.write("This chatbot uses BERT to understand user input and respond to weather-related questions.")

    # Button to reset chat history
    if st.button("Reset chat history"):
        st.session_state.history = []  # Clear history

    # Initialize chat history if not already set
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Create two columns layout
    col1, col2 = st.columns([1, 2])

    # Left column for user input
    with col1:
        user_input = st.text_input("Your message:", key="user_input")

    # Right column for chat history
    
    with col2:
        # Display chat history
        st.markdown("### Chat History")   
        
        # Greet the user and ask about the user's name 
        st.write(f"**Chatbot:** Hello! I'm a weather-chatbot! Can you give me your name?")
           
        if user_input:
            st.session_state.history.append(f"**User:** {user_input}")

            # Extract name if not already done
            if not any("Nice to meet you" in msg for msg in st.session_state.history):
                # Get the user's name from the user's answer
                name = model.extract_name(user_input)
                if name:
                    rep = f"**Chatbot:** Nice to meet you, {name}! What would you like to know about the weather today?"
                    
                    st.session_state.history.append(rep)
                else:
                    rep = "**Chatbot:** I couldn't catch your name. Please enter it again."
                    
                    st.session_state.history.append(rep)
            else:
                # Find the question that most closely matches the user input
                idx = model.find_similar_question(user_input, predef_emb_path)
                
                # The first 7 predefined questions in the predefined_input.txt file are 7 weather questions, the rest are goodbye questions.
                if idx < 7:  
                    rep = f"**Chatbot:** {predef_responses[idx]}"                    
                else:
                    rep = f"**Chatbot:** Goodbye! Have a great day!"
                st.session_state.history.append(rep)
                    
            user_input = None
        
        for message in st.session_state.history:
            st.write(message)  
            
if __name__ == "__main__":
    main()
