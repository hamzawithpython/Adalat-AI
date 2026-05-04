import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas.legal_response import (
    Citation, RightsRecord, Language,
    Jurisdiction, build_legal_response
)


def test_citation_model():
    c = Citation(source="uk_tenant_fees_act.pdf", page=5,
                 jurisdiction="UK", relevance_score=0.75)
    assert c.relevance_score == 0.75


def test_rights_record_model():
    r = RightsRecord(right="Right to deposit return",
                     legal_basis="UK Tenant Fees Act 2019, Section 12",
                     recourse="File claim at county court")
    assert r.deadline is None


def test_build_legal_response():
    mock_result = {
        "query": "What are my rights?",
        "translated_query": None,
        "language": "english",
        "jurisdiction": "UK",
        "answer": "You have rights under the Tenant Fees Act.",
        "citations": [{"source": "uk_tenant_fees_act.pdf",
                       "page": 5, "jurisdiction": "UK",
                       "relevance_score": 0.75}]
    }
    mock_rights = [{"right": "No prohibited fees",
                    "legal_basis": "Section 1, Tenant Fees Act",
                    "recourse": "Report to local authority"}]
    response = build_legal_response(mock_result, mock_rights)
    assert response.schema_valid == True
    assert response.confidence == 0.75


def test_jurisdiction_enum():
    assert Jurisdiction.UK == "UK"
    assert Jurisdiction.PK == "PK"
    assert Jurisdiction.DE == "DE"