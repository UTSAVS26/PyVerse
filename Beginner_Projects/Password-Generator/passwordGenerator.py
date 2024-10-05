import streamlit as st
import random
import string
import pyperclip

def passwordGenerator():
    def generate_password(length, use_upper, use_lower, use_digits, use_special):
        characters = ''
        if use_upper:
            characters += string.ascii_uppercase
        if use_lower:
            characters += string.ascii_lowercase
        if use_digits:
            characters += string.digits
        if use_special:
            characters += string.punctuation

        if characters == '':
            st.error("Please select at least one character type!")
            return ""

        password = ''.join(random.choice(characters) for _ in range(length))
        return password

    # app
    st.title("Password Generator")

    with st.form("password_generator_form"):
        length = st.slider("Password length", min_value=4, max_value=30, value=8)
        use_upper = st.checkbox("Include A-Z", value=True)
        use_lower = st.checkbox("Include a-z", value=True)
        use_digits = st.checkbox("Include 0-9", value=True)
        use_special = st.checkbox("Include special characters", value=True)

        submitted = st.form_submit_button("Generate Password")
        
        if submitted:
            password = generate_password(length, use_upper, use_lower, use_digits, use_special)
            st.session_state.generated_password = password

    if "generated_password" in st.session_state and st.session_state.generated_password:
        st.write("Generated Password:")
        st.code(st.session_state.generated_password, language="text")
        if st.button("Copy to Clipboard"):
            pyperclip.copy(st.session_state.generated_password)
            st.success("Password copied to clipboard!")
passwordGenerator()