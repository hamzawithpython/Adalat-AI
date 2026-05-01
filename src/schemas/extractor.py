import os
import sys
import json
import re
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EXTRACT_PROMPT = ChatPromptTemplate.from_template("""
You are a legal claim extractor. Extract structured rights from this legal answer.

LEGAL ANSWER:
{answer}

Extract ALL rights mentioned. Return ONLY a valid JSON array like this:
[
  {{
    "right": "specific right the person has",
    "legal_basis": "Article/Section number and document name",
    "obligation": "what the other party must do (or null)",
    "deadline": "time limit if mentioned (or null)",
    "recourse": "what action the user can take"
  }}
]

Rules:
- Return ONLY the JSON array, no other text
- If no clear rights found, return []
- Keep each field concise (max 2 sentences)
- legal_basis must reference specific article/section
""")


def extract_rights(answer: str) -> list[dict]:
    """Extract structured rights from LLM answer."""
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024
    )

    chain = EXTRACT_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({"answer": answer})
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            rights = json.loads(match.group())
            logger.info(f"Extracted {len(rights)} rights records")
            return rights
        return []
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return []