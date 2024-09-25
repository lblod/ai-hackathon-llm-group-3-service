""" This Python module contains code to use LLM's to read and analyse legal documentation pertaining to
Onroerend Erfgoed. Files can be parsed, read, analysed, and summarized, and advice can be formulated based on
a user query.

For example a user can ask "Can I change the windows?" then, based on provided documentation, the LLM will read the
document, and formulate an advice based on what's found in the documentation.

A main guard is implemented meaning the module can therefore be ran directly for testing.
"""
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
# from langchain_openai import AzureChatOpenAI
from hf_local_llm import LocalHFLLM

import requests

# OpenAI api key (will be deactivated after the hackathon, resource cost is monitored)
load_dotenv()


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def setup_llm() -> LocalHFLLM:
    """ Loads LLM locally """
    return LocalHFLLM("BramVanroy/fietje-2-instruct")


def get_hf_reply(user_prompt, system_prompt, context):
    prompt = f"<|system|>{system_prompt}<|user|>{user_prompt}\ncontext:{context}<|im_start|>assistant"

    # Payload for the API call
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL_HF, headers=HF_HEADERS, json=payload)
    return response.json()


def str_to_doc(
        content: str,
        metadata: dict
) -> Document:
    """ Small helper function to convert text to a Langchain document. """

    return Document(page_content=content, metadata=metadata)


def parse_pdf(
        pdf_path: Path,
        page_break="<div style='break-before:page'></div>"
) -> Document:
    """ Parses a given PDF file and returns its content as a Langchain Document instance. """

    # Make sure we are dealing with a pdf
    if not pdf_path.suffix == ".pdf":
        raise IOError("Parsing is only supported for pdf files for now...")

    # Load the pdf doc
    pdf_document = fitz.open(pdf_path)
    content = ""

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        try:
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            content += f"{page_break}Page {page_num}\n\n" + text + "\n"
        except Exception as e:
            logger.warning(f"There was an issue parsing {pdf_document.name} page "
                           f"{page_num} and it will be skipped. The error was: {e}")

    # Return the langchain document
    return str_to_doc(content, metadata={"reference": pdf_path.name})


def summarize_documents(
        legal_docs: List[Document],
        page_break="<div style='break-before:page'></div>"
) -> List[Document]:
    """ Makes a summary of what is and what isn't allowed in terms of works. """

    prompt = """ In what follows you will be provided with legal documentation pertaining to a specific 
    building, monument, site, or similar property. The documents will be provided in dutch. Your task is to thoroughly analyze the document, focussing in 
    particular on maintenance works, embellishment, modifications, or enhancements that can be carried out on 
    the property. Ignore historical restaurations and distructions.

    Once you have understood the whole document you must return:
    - allowed works: A list of allowed works for which you need no permit
    - restrictions: A list of restrictions and obligations that apply to any works to be carried ont on the property
    - forbidden works: A list of things you are not allowed to do on the property without formal approval
    - a clear message formed as "No relevant passages identified" in case nothing is mentioned about the above.
    
    Make sure to always reply with Markdown and adhere to the following format:
    ### Allowed Works:
    1. **Allowed 1**: Description
    2. **Allowed 2**: Description
    3. **Allowed 3**: Description
    
    ### Restrictions:
    1. **Restriction 1**: Description
    2. **Restriction 2**: Description
    
    ### Forbidden Works:
    1. **Forbidden 1**: Description
    2. **Forbidden 2**: Description
    3. **Forbidden 3**: Description
    4. **Forbidden 4**: Description
    
    One example output could be:
    ### Allowed Works:
    1. **Routine Maintenance**: Regular upkeep and minor repairs that do not alter the structure or appearance of the property.
    2. **Approved Works**: Any works that have been pre-approved as part of the management plan, as listed in the annex of the plan.
    3. **Emergency Repairs**: Immediate repairs necessary to prevent further damage to the property, provided they align with the management plan.
    
    ### Restrictions:
    1. **Management Plan Adherence**: All works must adhere to the stipulations and guidelines provided in the approved management plan.
    2. **Notification of Stakeholders**: If multiple rights holders or users were involved in creating the management plan, they must be notified of its approval as soon as possible.
    
    ### Forbidden Works:
    1. **Unapproved Alterations**: Any modifications, embellishments, or enhancements not listed in the approved management plan require formal approval.
    2. **Major Structural Changes**: Major structural changes or any works that significantly alter the appearance or integrity of the property without formal approval.
    3. **Unauthorized Cultural Goods Handling**: Any handling or movement of cultural goods not listed in the approved annex of the management plan requires formal approval.
    4. **Non-compliant Works**: Any works that do not comply with the guidelines and requirements laid out in the management plan.

    If you did not find any relevant information for whatever reason you can simply reply with "No relevant passages identified"
    """

    # Inner function to process a single page
    def _process(page, doc):
        reply = llm.request(prompt, page)

        if reply.content.lower().find("no relevant passages") == -1:
            origin = doc.metadata["reference"] + " " + page.split("\n")[0]
            logger.debug(f"Relevant mentions found on {origin}")
            return Document(
                page_content=reply.content,
                metadata={"reference": origin}
            )
        return None

    # Configure the llm using Azure OpenAI for now...
    llm = setup_llm()

    # Analyse the relevant documents one by one, page by page
    relevant_docs = []
    for doc_nr, doc in enumerate(legal_docs):
        content = doc.page_content
        pages = content.split(page_break)

        # Parallel execution for speed
        with ThreadPoolExecutor() as executor:
            pool = {executor.submit(_process, page, doc): page for page in pages}
            for future in as_completed(pool):
                result = future.result()
                if result:
                    relevant_docs.append(result)

    return relevant_docs


def analyse_documents(
        legal_docs: List[Document],
        work_query: str
) -> Document:
    """ Analyses a list of Langchain documents, compares to a user query, and generates specific advice. """

    prompt = """ You will be provided with legal documentation pertaining to a specific building, monument, site, or 
    similar property. A user will also provide you with a query about certain works or modifications to be done to the 
    property. Your task is to fully understand what implications the works to be done have on the property and 
    to check the documentation whether what the user wants to do is allowed or not.

    If the works that the user wants to perform are implicitly or explicitly mentioned or covered in the 
    documentation then you must reply with the relevant piece of information.

    If the works that the user wants to perform are not at all covered by the documentation then 
    you can simply reply with "No relevant passages identified."
    """

    # Formulate advice based on content and query
    relevant_content = "\n".join([rd.page_content for rd in legal_docs])

    reply = get_hf_reply(user_prompt=work_query,
                         system_prompt=prompt,
                         context=relevant_content)

    print(reply)
    return Document(page_content=reply[0]["generated_text"])


def run(
        documents: List[Document],
        query: str,
        page_break="<div style='break-before:page'></div>"
) -> str:
    """ This function is the interface between the microservice implementation code and the AI code.

    Args:
        documents: List of Langchain documents. Each Langchain document needs to have a 'reference' field in the
            metadata. If the document pertains to a multi-page document then pages must be separated in text with
            the specified page break
        query: The user query
        page_break: A string to be inserted into the document's page content at the end of each page.

    Returns:
        An AI response string
    """

    # Input validation
    if not isinstance(documents, List) or any([not isinstance(d, Document) for d in documents]):
        raise TypeError("Documents is supposed to be a list of Langchain documents")

    # Summarize
    summary = summarize_documents(documents, page_break=page_break)

    # Analyze
    analysis = analyse_documents(summary, work_query=query)

    # Return as a string
    return analysis.page_content


def _get_work_query() -> str:
    """ Get a random work query for testing. """

    random_selector = random.randint(1, 7)
    # random_selector = 6
    if random_selector == 1:
        return "Ik wil mijn gevel isoleren"
    elif random_selector == 2:
        return "Mag ik nieuwe ramen zetten"
    elif random_selector == 3:
        return "Ik wil de tuin opnieuw aanleggen"
    elif random_selector == 4:
        return "Die lelijke schoorsteen moet weg"
    elif random_selector == 5:
        return "Ik wil het interieur opfrissen"
    elif random_selector == 6:
        return "Ik wil de muren herplijsteren"
    else:
        return "Ik wil het hele gebouw slopen"


def _get_approved_beheersplan_docs() -> List[Document]:
    """ Get random beheersplannen for testing. """

    # Select a random project for testing purposes
    random_selector = random.randint(1, 1)
    if random_selector == 1:
        base_path = Path("data/beheersplannen/parochiekerk")
    else:
        raise ValueError("Random selector out of range")

    # See if we can find an approval
    approved = False
    for file in base_path.iterdir():
        if (file.stem.lower().find("beheer") != -1 and
                file.stem.lower().find("plan") != -1 and
                file.stem.lower().find("goed") != -1 and
                file.stem.lower().find("keur") != -1):
            logger.info(f"Approval document: {file}")
            approved = True

    # If approved parse relevant docs
    if approved:
        relevant_docs = []
        for f in base_path.iterdir():
            if f.stem.lower().find("beheer") != -1 and f.stem.lower().find("plan") != -1:
                relevant_docs.append(parse_pdf(f))
        return relevant_docs

    # Else return simply None (not the same as an empty list)
    else:
        return None


def _demo() -> None:
    """ This function is here just for demo/testing purpose and shows what the pipeline could look like applied on
    one directory containing the 'beheersplannen' for one OE.

    This requires having the necessary data available. For now we decided to push 1 example pdf to Github.
    TODO: PRIO 3 Don't push pdf's to Git, but download them automatically locally. """

    beheersplan_docs = _get_approved_beheersplan_docs()
    work_query = _get_work_query()

    # An approved beheersplan was found
    if beheersplan_docs is not None:
        logger.info(f"Work query is: {work_query}")
        logger.info("Relevant beheersplannen:")
        for d in beheersplan_docs:
            logger.info("-> " + d.metadata["reference"])

        # Run AI stack
        result = run(beheersplan_docs, work_query)
        logger.info(result)

    # No approved beheersplan was found
    else:
        logger.info(f"No approved beheersplan found")


if __name__ == "__main__":
    _demo()
