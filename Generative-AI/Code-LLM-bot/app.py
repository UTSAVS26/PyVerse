import streamlit as st
from groq import Groq
# from streamlit_extras.badges import badge

def github_badge():
    badge(type="github", name="yashksaini-coder/Code-Whisper")
    
def twitter_badge():
    badge(type="twitter", name="EasycodesDev")

def groq_completions(user_content, model, api_key):
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI-powered coding assistant here to help with programming challenges. \n"
                        "You can assist with various tasks, including:\n\n"
                        "- **Debugging Code:** Identify and fix errors in code shared by the user.\n"
                        "- **Explaining Concepts:** Provide detailed explanations of programming concepts.\n"
                        "- **Code Suggestions:** Offer code snippets and suggest approaches to implement features.\n"
                        "- **Optimization Tips:** Advise on improving code performance.\n"
                        "- **Learning Resources:** Recommend tutorials, articles, and other resources to help the user learn something new."
                    )
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=0.5,
            max_tokens=5640,
            top_p=1,
            stream=True,
            stop=None,
        )

        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""

        return result

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Streamlit interface
def main():
    st.set_page_config(page_title="AI Assistant", page_icon="images/logo.png")
    st.title("AI Coding Assistant")
    with st.sidebar:
        st.header("API Key Setup")
        api_key = st.text_input("Enter your GROQ API Key", type="password")
        try:
            if api_key:
                client = Groq(api_key=api_key)
                completion = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "system", "content": "Test"}],
                    temperature=0.5,
                    max_tokens=5,
                    top_p=1,
                    stream=True,
                    stop=None,
                )
                
                result = ""
                for chunk in completion:
                    result += chunk.choices[0].delta.content or ""

                if api_key.startswith('gsk-'):
                    st.warning("Please enter your Groq API key!", icon='⚠')
                else:
                    st.sidebar.success("API Key is valid!")
            else:
                st.warning("Please enter and validate your API Key in the sidebar.")
                
        except Exception:
            st.sidebar.error("Server is unreachable")
             
    model_options = [
        "mixtral-8x7b-32768",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "gemma2-9b-it",
        "gemma-7b-it",
    ]
    selected_model = st.selectbox("Select Model", model_options)
    user_content = st.text_input("How can I help you today?", key="query_input")  # Added session key for reset
    if st.button("Submit"):
        if not user_content:
            st.warning("Please enter your query to proceed.")
            return

        if not api_key:
            st.warning("Please enter and validate your API Key in the sidebar.")
            return
        
        answer = groq_completions(user_content, selected_model, api_key)
        if answer:
            st.success("Response generated successfully!")
            st.text_area("", value=answer, height=min(len(answer) * 20, 500), max_chars=None, key=None)
        else:
            st.error("Failed to generate a response. Please try again.")

    # Clear button functionality added with session state reset
    if st.button("Clear"):
        st.session_state["query_input"] = ""  # Resets the input field

if __name__ == "__main__":
    main()
