import os
import re
import argparse
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="client_secrets.json"
from flask import Flask, jsonify, request
from google.cloud import vision, language
from google.cloud import speech_v1p1beta1 as speech
from typing import Sequence
from pydub import AudioSegment

from config import SERVER_URL, SERVER_PORT

if not os.path.exists('uploads'):
    os.makedirs('uploads')

app = Flask(__name__)

def analyze_image_from_path(
    image_path: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response

def analyze_text_entities(text: str) -> language.AnalyzeEntitiesResponse:
    client = language.LanguageServiceClient()
    document = language.Document(
        content=text,
        type_=language.Document.Type.PLAIN_TEXT,
    )
    return client.analyze_entities(document=document)

def speech_to_text(
    config: speech.RecognitionConfig,
    audio: speech.RecognitionAudio,
) -> speech.RecognizeResponse:
    client = speech.SpeechClient()

    # Synchronous speech recognition request
    response = client.long_running_recognize(config=config, audio=audio).result(timeout=300)

    return response

def extract_email(string):
    email_regex = r"[a-zA-Z0-9\.\-+_]+@[a-zA-Z0-9\.\-+_]+\.[a-zA-Z]+"
    matches = remove_by_regex(string, email_regex)['matches']
    return matches
def extract_contract(string):
    contact_regex = "(^\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}$)|(\(\d{3}\)\s\d{3}-\d{4})|(\d{5}\s\d{5})"
    matches = remove_by_regex(string, contact_regex)['matches']
    return matches
def extract_site_url(string):
    site_url_regex = "((?:www\.)?[\w-]+(?<!gmail|yahoo)\.(?:com))(?=[^\w@]|$)"
    matches = remove_by_regex(string, site_url_regex)['matches']
    return matches
def remove_by_regex(text, regex):
    matches = []
    cleaned_text = []
    lines = text.split('\n')

    for line in lines:
        hits = re.findall(regex, line)
        if hits:
            for hit in hits:
                matches.append(hit)
        else:
            cleaned_text.append(line)

    cleaned_text = '\n'.join(cleaned_text)
    return {'matches': matches, 'cleanedText': cleaned_text}

@app.route("/")
def hello():
    return jsonify({"message": "Hello world"})

@app.route("/interpret", methods=['POST'])
async def interpretImage():
    file = request.files['image']
    filename = file.filename
    file_path = 'uploads/' + filename
    file.save(file_path)

    features = [vision.Feature.Type.TEXT_DETECTION]
    requiredEntities = {"ORGANIZATION": [], "PERSON": [], "ADDRESS": []};

    response = analyze_image_from_path(file_path, features)

    annotation = response.text_annotations[0];
    text = annotation.description if annotation else ''
    languageResults = analyze_text_entities(text)
    entities = languageResults.entities
    for entity in entities:
        type = language.Entity.Type(entity.type_).name
        if type in requiredEntities:
            requiredEntities[type] = [*requiredEntities[type], entity.name]
    email = extract_email(text)
    phone_number = extract_contract(text)
    site_url = extract_site_url(text)
    return jsonify({"email": email, "phone": phone_number, "site": site_url, **requiredEntities})

@app.route("/convert-voice-to-text", methods=['POST'])
async def convertVoiceToText():
    file = request.files['audio']
    filename = file.filename
    file_path = 'uploads/' + filename

    file.save(file_path)

    sound = AudioSegment.from_file(file_path, 'm4a')
    mono_sound = sound.set_channels(1)
    mono_sound.export(file_path[:-4] + '.wav', 'wav')

    file_path = file_path[:-4] + '.wav'

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US',
        alternative_language_codes=['th-TH', 'ar-DZ'],
        enable_automatic_punctuation=True,
    )

    response = speech_to_text(config, audio)
    transcript_builder = []
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        transcript_builder.append(alternative.transcript)

    transcript = "".join(transcript_builder)

    return jsonify({"text": transcript})

if __name__ == '__main__':
    app.run(SERVER_URL, SERVER_PORT)
