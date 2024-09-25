import requests
from flask import request, make_response
import io
from pypdf import PdfReader
import re
from escape_helpers import sparql_escape_uri, sparql_escape_string
from helpers import generate_uuid, query, update, logger
from string import Template
from .scripts.poc_beheersplan import run
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import fitz
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

def get_decisions(designation_object_id):
    query_string = Template('''
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX eli: <http://data.europa.eu/eli/ontology#>
        PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT ?identifier WHERE {
            ?designation mu:uuid $designation_object_id .
            ?designation ^eli:cites ?decision .
            ?decision dct:identifier ?identifier .
        }
    '''
    ).substitute(
        designation_object_id=sparql_escape_string(designation_object_id),
    )

    response = query(query_string)

    return response['results']['bindings'][0]['identifier']['value']

def get_plans(designation_object_id):
    query_string = Template('''
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX eli: <http://data.europa.eu/eli/ontology#>
        PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT ?identifier WHERE {
            ?designation mu:uuid $designation_object_id .
            ?designation ext:hasPlan ?plan .
            ?decision dct:identifier ?identifier .
        }
    ''',
        designation_object_id=sparql_escape_string(designation_object_id),
    )

    response = query(query_string)

    return response


def str_to_doc(
        content: str,
        metadata: dict
) -> Document:
    """ Small helper function to convert text to a Langchain document. """

    return Document(page_content=content, metadata=metadata)


@app.route('/designation-objects/<designation_object_id>/advice')
def advice(designation_object_id):
    query = request.args.get('query')
    decision_id = get_decisions(designation_object_id)
    decision_text = get_decision_text(decision_id)
    decision_doc = str_to_doc(decision_text, { 'reference': ''})

    return run([decision_doc], query, page_break='[[--PAGE BREAK--]]')
    return ''


def get_file_content(file_uri):
    response = requests.get(file_uri)
    response.raise_for_status()  
    
    return io.BytesIO(response.content)


def pdf_to_str(content):
    reader = PdfReader(content)
    all_lines = []

    for page in reader.pages:
        text = page.extract_text()
        for line in text.split('\n'):
            if not re.match(r"^\s{0,}(Pagina\s\d+\svan\s\d+\s{0,})?$", line):
                all_lines.append(line.strip())
    
    one_string = '[[--PAGE BREAK--]]'.join(all_lines)

    return one_string

def get_decision_text(decision_id):
    decision_uri = f'https://besluiten.onroerenderfgoed.be/besluiten/{decision_id}'
    decision_file_uri = get_decision_file(decision_uri)
    content = get_file_content(decision_file_uri)
    str_content = pdf_to_str(content)
    
    return str_content

def get_plan_text(plan_id):
    decision_uri = f'https://plannen.onroerenderfgoed.be/plannen/{decision_id}'
    decision_file_uri = get_decision_file(decision_uri)
    content = get_file_content(decision_file_uri)
    str_content = pdf_to_str(content)
    
    return str_content


def get_decision_file(besluit_uri):
    response = requests.get(besluit_uri, headers={
        'Accept': 'application/json'
    }).json()

    files = [
        f'{besluit_uri}/bestanden/{bestand["id"]}'
        for bestand in response['bestanden']
        if (
            bestand['bestandssoort']['soort'] == 'Besluit'
            and not bestand['naam'].endswith('metcert.pdf')
        )
    ]

    return files[0]

def get_plan_file(besluit_uri):
    response = requests.get(besluit_uri, headers={
        'Accept': 'application/json'
    }).json()

    files = [
        f'{besluit_uri}/bestanden/{bestand["id"]}'
        for bestand in response['bestanden']
        if (
            bestand['bestandssoort_id'] == 1
        )
    ]

    return files[0]


# Here we should check if AanduidingsObject is inserted
# Add AI generated toelatingsplichtige handelingen here
@app.route('/delta')
def delta():
    data = request.get_json(force=True)
