import os
import PyPDF2
import tiktoken
from flask import Flask, request, jsonify

from dotenv import load_dotenv
import requests
import json


load_dotenv()
# Set up the Hugging Face Inference API parameters
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": "Bearer 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"}


def _get_llm_response(user_input):
    system_prompt = "Je beantwoordt de gestelde vraag in de gegeven context. vertaal in het nederlands"

    context = """
    ingreep in de bodem: elke wijziging van de eigenschappen van de ondergrond door de verwijdering of toevoeging van materie,
    de verhoging of verlaging van de grondwatertafel, of het samendrukken van de materialen waaruit de ondergrond bestaat.
    Voor de berekening van de totale oppervlakte van de ingreep in de bodem wordt rekening gehouden met de oppervlakte
    van de vergunningsplichtige werken of handelingen zoals opgenomen in de vergunningsaanvraag;
    """

    # Get user input for the user prompt
    # user_prompt = "Waarmee wordt rekening gehouden bij het berekenen van de totale oppervlakte van de ingreep in de grond?"
    user_prompt = input("Please enter your question: ")

    # Combine the messages into a single prompt
    # This structure is specific to BramVanroy/fietje-2-instruct model
    prompt = f"<|system|>{system_prompt}<|user|>{user_prompt}\ncontext:{context}<|im_start|>assistant"

    # Payload for the API call
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Initialize the Flask app
app = Flask(__name__)

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({'error': 'No user_input provided'}), 400

    response = _get_llm_response(user_input)

    return jsonify({'response': response[0]["generated_text"]})


if __name__ == '__main__':
    app.run(debug=True)
