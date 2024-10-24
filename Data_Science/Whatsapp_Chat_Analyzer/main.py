import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import emoji
from collections import Counter
import plotly.graph_objs as go
import numpy as np
import requests
import re
import zipfile
import io
from datetime import datetime

# Function to handle date conversion
def handle_date(date_str):
    """
    Converts a date string to datetime or timedelta.
    
    Args:
    date_str (str): Date string to convert
    
    Returns:
    datetime or timedelta: Converted date/time object
    """
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        # Handle incompatible format (e.g., "0 days 00:04:00")
        time_part = date_str.split()[2]
        return pd.to_timedelta(time_part)

# Function to convert 24-hour time to 12-hour format
def twelve_hr_convert(time_str):
    """
    Converts 24-hour time format to 12-hour format.
    
    Args:
    time_str (str): Time string in 24-hour format
    
    Returns:
    tuple: (time_string, am/pm indicator)
    """
    return datetime.strptime(time_str, "%H:%M").strftime("%I:%M %p")

# Function to process chat file
@st.cache_data(ttl=300)  # Added TTL for cache expiration
def process_chat_file(file_contents):
    """
    Processes the chat file contents and extracts relevant information.
    
    Args:
    file_contents (str): Contents of the chat file
    
    Returns:
    tuple: (full_df, message_df, emoji_df, emoji_author_df)
    """
    # Combined regex for different date/time formats
    pattern = r"(\d+\/\d+\/\d+), (\d+:\d+(?:\s?[ap]m)?) \- ([^\:]*):(.*)"
    data = []

    # Process each line of the file
    for line in file_contents.split("\n"):
        match = re.match(pattern, line)
        if match:
            groups = match.groups()
            date, time = groups[0], groups[1]

            # Remove any invisible or non-breaking spaces from time strings
            time = re.sub(r'[\u202f\u00a0]', '', time.strip())
            
            ampm, author, message = None, groups[2], groups[3]
            if 'am' in time.lower() or 'pm' in time.lower():
                ampm = time[-2:].lower()  # Extract am/pm indicator
                time = time[:-2].strip()  # Remove am/pm from time string

            if not ampm:  # Convert to 12-hour format if no am/pm
                time, ampm = twelve_hr_convert(time)

            data.append({
                "Date": date,
                "Time": time,
                "AM/PM": ampm,
                "Author": author.strip(),
                "Message": message.strip()
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Extract emojis from messages (using emoji package for accuracy)
    df["Emoji"] = df["Message"].apply(lambda text: emoji.emoji_list(text))

    # Remove media and null messages
    message_df = df[~df["Message"].isin(['<Media omitted>', 'null'])]

    # Convert date and time columns
    message_df["Date"] = pd.to_datetime(message_df["Date"])

    # Convert time, accounting for am/pm format
    message_df["Time"] = pd.to_datetime(message_df["Time"] + ' ' + message_df["AM/PM"], format='%I:%M %p', errors='coerce').dt.strftime('%H:%M')

    # Calculate letter and word counts using vectorized operations
    message_df['Letter_Count'] = message_df['Message'].str.len()
    message_df['Word_count'] = message_df['Message'].str.split().str.len()

    # Process emojis
    total_emojis_list = [emoji['emoji'] for emojis in message_df['Emoji'] for emoji in emojis]
    emoji_author_counts = {author: Counter() for author in message_df['Author'].unique()}
    for emojis, author in zip(message_df['Emoji'], message_df['Author']):
        for emoji_data in emojis:
            emoji_author_counts[author][emoji_data['emoji']] += 1

    emoji_author_df = pd.DataFrame.from_dict(emoji_author_counts, orient='index').fillna(0)
    emoji_df = pd.DataFrame(sorted(Counter(total_emojis_list).items(), key=lambda x: x[1], reverse=True),
                            columns=["emoji", "count"])

    # Combine date and time into DateTime column
    message_df['DateTime'] = pd.to_datetime(message_df['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
                                            message_df['Time'], errors='coerce')
    message_df = message_df.sort_values(by='DateTime')

    # Calculate response times
    message_df['Response Time'] = pd.NaT
    last_message_time = {}
    for index, row in message_df.iterrows():
        author, current_time = row['Author'], row['DateTime']
        if author in last_message_time:
            last_time = last_message_time[author]
            if last_time is not pd.NaT:
                message_df.at[index, 'Response Time'] = (current_time - last_time).total_seconds()
        last_message_time[author] = current_time

    return df, message_df, emoji_df, emoji_author_df

# Function to extract text file from zip archive
def extract_text_file(uploaded_file):
    """
    Extracts a text file from a zip archive.
    
    Args:
    uploaded_file (UploadedFile): The uploaded zip file
    
    Returns:
    str: Contents of the extracted text file or an error message
    """
    try:
        zip_file = zipfile.ZipFile(io.BytesIO(uploaded_file.read()))
        text_file_name = next((name for name in zip_file.namelist() if name.endswith(".txt")), None)
        if text_file_name:
            return zip_file.read(text_file_name).decode("utf-8")
        else:
            return "No text file found in the zip archive."
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Main Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config("WhatsApp Chat Analyzer", page_icon="📲", layout="centered")
    st.title("Chat Data Visualization")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a chat file")
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "txt":
            file_contents = uploaded_file.read().decode("utf-8")
        elif file_extension == "zip":
            file_contents = extract_text_file(uploaded_file)
        else:
            st.error("Please upload a .txt or .zip file")
            return

        # Process chat data
        df, message_df, emoji_df, emoji_author_df = process_chat_file(file_contents)

        # Layout enhancement with columns and expanders
        with st.expander("Basic Information (First 20 Conversations)"):
            st.write(df.head(20))

        with st.expander("Author Stats"):
            for author in message_df['Author'].unique():
                req_df = message_df[message_df["Author"] == author]
                st.subheader(f'Stats of {author}:')
                st.write(f'Messages sent: {req_df.shape[0]}')
                words_per_message = (np.sum(req_df['Word_count'])) / req_df.shape[0]
                st.write(f"Words per message: {words_per_message:.2f}")
                emoji_count = req_df['Emoji'].apply(len).sum()
                st.write(f'Emojis sent: {emoji_count}')
                avg_response_time = round(req_df['Response Time'].mean(), 2)
                st.write(f'Average Response Time: {avg_response_time} seconds')

        # Display emoji distribution
        with st.expander("Emoji Distribution"):
            fig = px.pie(emoji_df, values='count', names='emoji', title='Emoji Distribution')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)
        
        # Display emoji usage by author
        with st.expander("Emoji Usage by Author"):
            fig = px.bar(emoji_author_df, x=emoji_author_df.index, y=emoji_author_df.columns, 
                         title="Emoji Usage by Author", barmode='stack')
            fig.update_layout(xaxis_title="Authors", yaxis_title="Count", legend_title="Emojis")
            st.plotly_chart(fig)

        # Display top 10 days with most messages
        with st.expander("Top 10 Days With Most Messages"):
            messages_per_day = message_df.groupby(message_df['DateTime'].dt.date).size().reset_index(name='Messages')
            top_days = messages_per_day.sort_values(by='Messages', ascending=False).head(10)
            fig = go.Figure(data=[go.Bar(
                x=top_days['DateTime'],
                y=top_days['Messages'],
                marker=dict(color='rgba(58, 71, 80, 0.6)', line=dict(color='rgba(58, 71, 80, 1.0)', width=1.5)),
                text=top_days['Messages']
            )])
            fig.update_layout(
                title='Top 10 Days with Most Messages',
                xaxis=dict(title='Date', tickfont=dict(size=14, color='rgb(107, 107, 107)')),
                yaxis=dict(title='Number of Messages', titlefont=dict(size=16, color='rgb(107, 107, 107)')),
                bargap=0.1,
                bargroupgap=0.1,
                paper_bgcolor='rgb(233, 233, 233)',
                plot_bgcolor='rgb(233, 233, 233)',
            )
            st.plotly_chart(fig)

        # Display message distribution by day
        with st.expander("Message Distribution by Day"):
            day_df = message_df[["Message", "DateTime"]].copy()
            day_df['day_of_week'] = day_df['DateTime'].dt.day_name()
            day_df["message_count"] = 1
            day_counts = day_df.groupby("day_of_week").size().reset_index(name='message_count')
            fig = px.line_polar(day_counts, r='message_count', theta='day_of_week', line_close=True)
            fig.update_traces(fill='toself')
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig)

        # Display word cloud
        with st.expander("Word Cloud"):
            text = " ".join(str(review) for review in message_df.Message)
            nltk.download('stopwords', quiet=True)
            stopwords = set(nltk.corpus.stopwords.words('english'))
            # Add custom stopwords here
            stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
            stopwords.update(list(set(stopwords_list.decode().splitlines())))
            # Create the Word Cloud
            wordcloud = WordCloud(width=800, height=400, random_state=1, background_color='white', colormap='Set2', collocations=False, stopwords=stopwords).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        # Display Creator Details
        html_temp = """
        <div style="text-align: center; font-size: 14px; padding: 5px;">
        Created by Aritro Saha - 
        <a href="https://aritro.tech/">Website</a>, 
        <a href="https://github.com/halcyon-past">GitHub</a>, 
        <a href="https://www.linkedin.com/in/aritro-saha/">LinkedIn</a>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

#driver code
if __name__ == "__main__":
    main()
