import os
import streamlit as st
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


# Streamlit UI
st.set_page_config(page_title='AI Data Science Tutor', page_icon="📊", layout = 'wide')

st.markdown("""
    <style>
        /* Main App Background */
        .stApp {
            background: linear-gradient(to right, #f5f7fa, #c3cfe2); /* Light gradient */
            color: #333;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #4b6cb7, #182848); /* Blue-purple gradient */
            padding: 20px;
            border-radius: 15px;
            color: white;
        }
        
        /* Sidebar Titles */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #fff;
            font-weight: bold;
        }

        /* Sidebar Text */
        [data-testid="stSidebar"] p {
            color: #f0f0f0;
            font-size: 16px;
        }

        /* Sidebar Bullet Points */
        [data-testid="stSidebar"] ul {
            color: #fff;
        }

        /* Chat Messages Styling */
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
            margin: 5px 0;
            background-color: #fff;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }

        /* Chat Input Box */
        .stChatInput {
            border: 2px solid #4a90e2;
            border-radius: 15px;
            padding: 10px;
            background-color: #ffffff;
        }
        
        /* Title Styling */
        h1 {
            text-align: center;
            font-family: 'Poppins', sans-serif;
            color: #1f2937;
            font-weight: 600;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(to right, #ff758c, #ff7eb3);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 8px 15px;
            font-size: 14px;
            font-weight: bold;
        }

        .stButton > button:hover {
            background: linear-gradient(to right, #ff7eb3, #ff758c);
        }

        /* Download Chat History - Black & Bold */
        .download-chat-history {
            font-weight: bold;
            color: black;
        }

    </style>
""", unsafe_allow_html=True)


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

chat_model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', temperature = 0.7, google_api_key=api_key)

# Initiliazing memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

## Sidebar - About the app
st.sidebar.title("📌 About this App")
st.sidebar.info("""
This AI powered tutor helps you learn Data Science interactively.
- **Ask any Data Science questions**
- **Select your learning level** (Beginner, Intermediate, Advanced)
- **Get clear, structured explanations**

**Enjoy learning with AI!** 🚀              
""")

## Sidebar - Options for learning level
st.sidebar.title("Settings")
user_level = st.sidebar.radio("Select your learning level:", ["Beginner", "Intermediate", "Advanced"])


system_message = SystemMessage(
    content = f"You are an AI tutor specialized in answering only Data Science-related questions. "
            f"If the user asks anything outside Data Science, politely refuse to answer. "
            f"Provide responses based on the user's learning level: {user_level}."
)


st.title("🔍 Ask AI: Your Data Science Helper")


def extract_response_for_level(full_response, level):
    """
    Extract only the relevant part of the AI response for the selected user level.
    """
    sections = {
        "Beginner": "🔰 Beginner:",
        "Intermediate" : "📚 Intermediate:",
        "Advanced" : "🚀 Advanced:"
    }

    if sections[level] in full_response:

        # Find where the section starts
        start_idx = full_response.find(sections[level])
        
        # Find where the next section starts (if any) and slice response
        next_section_idx = min(
            [full_response.find(sec) for sec in sections.values() if full_response.find(sec) > start_idx] + [len(full_response)]
        )
        
        return full_response[start_idx:next_section_idx].strip()
    
    return full_response  # If no structured response is detected, return as-is.


# Dislaying chat history
chat_history = []
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {msg.content}")


# User input
user_query = st.chat_input("Ask me anything about Data Science...")

if user_query:
    conversation_history = [system_message] + st.session_state.memory.chat_memory.messages + [HumanMessage(content=user_query)]

    ai_response = chat_model.invoke(conversation_history)

    filtered_response = extract_response_for_level(ai_response.content, user_level)


    st.session_state.memory.chat_memory.add_user_message(user_query)
    st.session_state.memory.chat_memory.add_ai_message(filtered_response)

    # Display user message
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_query}")

     # Display AI Response
    with st.chat_message("assistant"):
        st.markdown(f"**AI:** {ai_response.content}")

    chat_history.append(f"**You:** {user_query}")
    chat_history.append(f"**AI:** {filtered_response}")

## Export CHat history as text file
if chat_history:
    chat_text = "\n".join(chat_history)
    st.sidebar.download_button("📥 ", chat_text, file_name = "chat_history.txt")
    

## Reset Chat Button
if st.button("Reset Chat"):
    st.session_state.memory.clear()
    st.rerun()