""" This script is a minimal test to:
- get a beheersplan
- parse a beheersplan
- formulate an advise based on the beheersplan and a user-defined query
"""
import logging
import os
import random
from pathlib import Path

import fitz
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm

# OpenAI api key (will be deactivated after the hackathon, resource cost is monitored)
load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# LLM prompts
RELEVANT_INFO_PROMPT = """

You will be provided with legal documentation pertaining to a specific building, monument, site, or similar property. Your task is to thoroughly analyze the document and determine the following:

Does the document mention any maintenance works, modifications, or enhancements that can be carried out on the property?
Are there any sections of the property specifically identified as protected within the document?
Does the document impose any restrictions or prohibitions with regards to the conservation of the property?

Your response should be a set of bullet points summarizing the information. If none of the above points are mentioned in the document, your response should simply be: "No relevant passages identified."

Example output:
- Maintenance works allowed: interior refurbishment, but color scheme must remain the same.
- Protected sections: none mentioned.
- Restrictions: changes to windows not allowed; all maintenance work must be reported to the authorities every 5 years.

You can add other bullet points here, but don't add bullet points that are not covered in the documentation.
"""

ANALYSIS_PROMPT = """ 

You will be provided with legal documentation pertaining to a specific building, monument, site, or similar property. 
A user will also provide you with a query about certain works or modifications to be done to the property.
Your task is to fully understand what implications the works to be done have on the property and 
to check the documentation whether what the user wants to do is allowed or not.

If the works that the user wants to perform are implicitly or explicitly mentioned or covered in the 
documentation then you must reply with the relevant piece of information.

If the works that teh user wants to perform are not at all covered by the documentation then 
you can simply reply with "No relevant passages identified."
"""


def setup_llm():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_VERSION,
    )


def get_work_query():
    """ Get a random work query for testing. """

    random_selector = random.randint(1, 7)
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
        return "Mag ik de dakgoot uitkuisen"
    else:
        return "Ik wil het hele gebouw slopen"


def get_approved_beheersplan_docs():
    """ Get random beheersplannen for testing. """

    # Select a random project for testing purposes
    # random_selector = random.randint(1, 3)
    random_selector = 3
    if random_selector == 1:
        base_path = Path("../data/beheersplannen/halle")
    elif random_selector == 2:
        base_path = Path("../data/beheersplannen/avhp")
    elif random_selector == 3:
        base_path = Path("../data/beheersplannen/parochiekerk")
    else:
        raise ValueError("Random selector out of range")

    # See if we can find an approval
    approved = False
    for file in base_path.iterdir():
        if (file.stem.lower().find("beheer") != -1 and
                file.stem.lower().find("plan") != -1 and
                file.stem.lower().find("goed") != -1 and
                file.stem.lower().find("keur") != -1):
            logger.debug(f"Approval document: {file}")
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


def parse_pdf(pdf_path):
    """ Straight-forward pdf parsing using Pymupdf. """

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
            content += f"[[--PAGE BREAK--]]Page {page_num}\n\n" + text + "\n"
        except Exception:
            pass

    # Create a LangChain Document
    langchain_document = Document(
        page_content=content,
        metadata={"reference": pdf_path.name}
    )

    # Return the langchain document
    return langchain_document


def summarize_relevant_passages(beheersplan_docs):
    """ Makes a summary of what is and what isn't allowed in terms of works. """

    # TODO: Async
    # Configure the llm using Azure OpenAI for now...
    llm = setup_llm()

    # Analyse page by page and look for relevant passages
    relevant_docs = []
    for doc_nr, doc in enumerate(beheersplan_docs):
        content = doc.page_content
        pages = content.split("[[--PAGE BREAK--]]")
        num_docs = len(beheersplan_docs)
        for page in tqdm(pages, leave=False, desc=f"Analysing doc {doc_nr + 1}/{num_docs}"):
            if len(page) < 10:
                continue
            messages = [SystemMessage(RELEVANT_INFO_PROMPT), HumanMessage(page)]
            reply = llm.invoke(messages)

            if reply.content.lower().find("no relevant passages") == -1:
                origin = doc.metadata["reference"] + " " + page.split("\n")[0]
                logger.debug(f"Relevant mentions found on {origin}")
                relevant_docs.append(Document(
                    page_content=reply.content,
                    metadata={"reference": origin}
                ))

    return relevant_docs


def analyse_relevant_passages(relevant_docs, work_query):
    # Configure the llm using Azure OpenAI for now...
    llm = setup_llm()

    # Formulate advice based on content and query
    relevant_content = "\n".join([rd.page_content for rd in relevant_docs])
    messages = [
        SystemMessage(ANALYSIS_PROMPT + relevant_content),
        HumanMessage(work_query)]
    reply = llm.invoke(messages)

    # TODO: add refs and argumentation
    return reply.content


def main():
    beheersplan_docs = get_approved_beheersplan_docs()

    # An approved beheersplan was found
    if beheersplan_docs is not None:
        logger.debug("Relevant beheersplannen:")
        for d in beheersplan_docs:
            logger.debug("-> " + d.metadata["reference"])

        # Get some dummy work querries to test
        work_query = get_work_query()
        logger.debug(f"Work query is: {work_query}")

        # Extract the relevant content
        relevant_docs = summarize_relevant_passages(beheersplan_docs)
        relevant_text = "\n".join([rd.page_content for rd in relevant_docs])
        logger.info(f"Summarized beheersplan:"
                    f"\n===========================================================\n"
                    f"{relevant_text}"
                    f"\n===========================================================\n")

        # Analyse the relevant content
        analysis_result = analyse_relevant_passages(relevant_docs, work_query)
        logger.info(f"Relevant passages based on user query {work_query}:"
                    f"\n===========================================================\n"
                    f"{analysis_result}"
                    f"\n===========================================================\n")

    # No approved beheersplan was found
    else:
        logger.info(f"No approved beheersplan found")


if __name__ == "__main__":
    main()
