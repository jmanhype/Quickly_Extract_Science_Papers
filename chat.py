import openai
import os
from time import sleep
from halo import Halo
import textwrap
import yaml
import torch
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import tempfile

# Set temporary directory
tempfile.tempdir = "./temp"

### File operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

### Sentiment analysis

def analyze_sentiment(response):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode_plus(response, return_tensors='pt')
    outputs = model(**inputs)
    sentiment_values, _ = torch.max(outputs.logits, dim=1)
    sentiment_label = sentiment_values.tolist()[0]
    return sentiment_label

### Context analysis

def get_context(response):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(response)
    entities = [ent.text for ent in doc.ents]
    return entities

### API functions

def chatbot(conversation, model="gpt-4-0613", temperature=0.7):
    max_retry = 7
    retry = 0
    while True:
        try:
            spinner = Halo(text='Thinking...', spinner='dots')
            spinner.start()

            response = openai.ChatCompletion.create(model=model, messages=conversation, temperature=temperature)
            text = response['choices'][0]['message']['content']

            spinner.stop()

            return text, response['usage']['total_tokens']
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = conversation.pop(0)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)

def chat_print(text):
    formatted_lines = [textwrap.fill(line, width=120, initial_indent='    ', subsequent_indent='    ') for line in text.split('\n')]
    formatted_text = '\n'.join(formatted_lines)
    print('\n\n\nCHATBOT:\n\n%s' % formatted_text)

    # Create a TTS object
    tts = gTTS(text=formatted_text, lang='en')

    # Save the speech audio into a file
    tts.save("response.mp3")

    # Load mp3 file with pydub
    audio = AudioSegment.from_mp3("response.mp3")

    # Play audio file
    play(audio)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(current_dir, 'key_openai.txt')
    openai.api_key = open_file(key_file).strip()

    paper = open_file('input.txt')
    if len(paper) > 22000:
        paper = paper[:22000]
    ALL_MESSAGES = [{'role': 'system', 'content': paper}]

    while True:
        # get user input
        user_input = input('\n\n\nUSER:\n\n')
        if not user_input.strip():
            # Empty submission, probably accidental
            continue
        ALL_MESSAGES.append({'role': 'user', 'content': user_input})

        # get response
        response, tokens = chatbot(ALL_MESSAGES)
        if tokens >= 7800:
            a = ALL_MESSAGES.pop(1)

        # Additional feature: Sentiment analysis
        sentiment = analyze_sentiment(response)
        print(f'Sentiment: {sentiment}')

        # Additional feature: Contextual understanding
        context = get_context(response)
        print(f'Context: {context}')

        chat_print(response)
        ALL_MESSAGES.append({'role': 'assistant', 'content': response})
