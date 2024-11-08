from transformers import pipeline
import spacy
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Example
text = """Your document or article content goes here. Make sure it's a large enough
passage to test the summarizer properly."""
print("Summary:", summarize_text(text))


nlp = spacy.load("en_core_web_sm")

def generate_flashcards(text):
    doc = nlp(text)
    flashcards = []
    for sent in doc.sents:
        # Extract main entities for Q&A pairs
        entities = [ent.text for ent in sent.ents]
        if entities:
            question = f"What is {entities[0]}?"
            answer = sent.text
            flashcards.append((question, answer))
    return flashcards

# Example
flashcards = generate_flashcards(text)
for q, a in flashcards:
    print("Q:", q)
    print("A:", a)



def generate_quiz_questions(text):
    questions = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        if "is" in sentence:
            question = sentence.replace("is", "is what")
            questions.append(question + "?")
    return questions

# Example
questions = generate_quiz_questions(text)
for q in questions:
    print("Quiz Question:", q)

def main():
    print("Welcome to the AI-Powered Study Assistant!")
    text = input("Enter the text you want to study: ")

    # Generate summary
    print("\nSummarizing text...")
    summary = summarize_text(text)
    print("Summary:", summary)

    # Generate flashcards
    print("\nGenerating flashcards...")
    flashcards = generate_flashcards(text)
    for i, (q, a) in enumerate(flashcards, 1):
        print(f"Flashcard {i} - Q: {q}")
        print(f"            A: {a}")

    # Generate quiz questions
    print("\nGenerating quiz questions...")
    questions = generate_quiz_questions(text)
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")

if __name__ == "__main__":
    main()
