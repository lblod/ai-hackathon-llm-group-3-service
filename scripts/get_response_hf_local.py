import os
import PyPDF2
import tiktoken
from flask import Flask, request, jsonify

from dotenv import load_dotenv
import requests
import json
from hf_local_llm import LocalHFLLM


load_dotenv()


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

    return llm.request_with_context(prompt, user_prompt, context)

# Initialize the Flask app
app = Flask(__name__)

llm = LocalHFLLM("BramVanroy/fietje-2-instruct")

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({'error': 'No user_input provided'}), 400

    response = _get_llm_response(llm, user_input)

    return jsonify({'response': response[0]["generated_text"]})


if __name__ == '__main__':
    app.run(debug=True)
