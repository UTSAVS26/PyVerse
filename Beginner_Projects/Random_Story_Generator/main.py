import random

# Story components expanded for longer story generation
sentence_starters = [
    "About 100 years ago,",
    "In the distant past,",
    "Once upon a time,",
    "Long ago, in a small village,"
]

characters = [
    "there lived a humble farmer named Jack.",
    "there was a kind old king named Henry.",
   "a curious young girl named Lily lived nearby.",
    "there dwelled a brave knight named Emma."
]

times = [
    "One full-moon night,",
    "On a cold winter evening,",
    "One bright summer morning,",
    "During a quiet autumn day,"
]

activities = [
   "they decided to take a stroll through the garden.",
   "they ventured out to explore the nearby forest.",
   "they were passing by the old mill.",
   "they were collecting herbs near the river."]
encounters = [
    "Suddenly, he saw a young lady standing alone under a tall oak tree.",
    "Out of nowhere, she noticed a mysterious glowing flower.",
    "He heard a soft whisper coming from the shadows.",
    "She found an old map hidden beneath some leaves."
]

descriptions = [
    "She seemed to be in her late 20s with a serene aura.",
    "The flower radiated a magical light unlike anything seen before.",
    "The whisper spoke of a secret treasure nearby.",
    "The map appeared ancient, with cryptic symbols covering it."
]

actions = [
    "Curious, he approached to offer help.",
    "Excited, she decided to follow the map’s directions.",
    "Carefully, he listened to the whisper and followed its guidance.",
    "Eagerly, she started searching for the treasure."
]

conclusions = [
    "After a long journey, they found a hidden meadow filled with glowing flowers.",
    "At the end of the path, they discovered an enchanted chest full of jewels.",
    "Together, they unlocked the mystery that had long been forgotten.",
    "Their bravery was rewarded with a blessing of prosperity and happiness."
]

endings = [
    "From that day forward, their lives were forever changed.",
    "And so, the village prospered thanks to their courage.",
    "Their story was told for generations as a tale of kindness and adventure.",
    "The magic of that night remained in the hearts of those who believed."
]

# Function to build a longer story with multiple sentences
def generate_long_story():
    story = ""
    story += random.choice(sentence_starters) + " "
    story += random.choice(characters) + " "
    story += random.choice(times) + " "
    story += random.choice(activities) + " "
    story += random.choice(encounters) + " "
    story += random.choice(descriptions) + " "
    story += random.choice(actions) + " "
    story += random.choice(conclusions) + " "
    story += random.choice(endings)
    return story

# Generate and print the story
print(generate_long_story())
