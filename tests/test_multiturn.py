"""Manual end-to-end test for multi-turn conversation memory.
Run with the FastAPI server already running on port 8000."""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def pretty_print_response(label: str, resp: dict):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print('='*70)
    print(f"session_id:        {resp.get('session_id', 'N/A')}")
    print(f"language:          {resp.get('language')}")
    print(f"jurisdiction:      {resp.get('jurisdiction')}")
    print(f"is_clarification:  {resp.get('is_clarification', False)}")
    print(f"confidence:        {resp.get('confidence')}")
    print(f"\nANSWER:\n{resp.get('answer', '')}")
    print(f"\nFOLLOW-UPS: {resp.get('follow_up_questions', [])}")


def main():
    # ── TURN 1 ────────────────────────────────────────────────────
    turn1_query = "mera landlord deposit wapas nahi de raha, mein Lahore mein hoon"
    print(f"\n→ Sending turn 1: {turn1_query!r}")

    r1 = requests.post(
        f"{BASE_URL}/ask",
        headers={"Content-Type": "application/json"},
        json={"query": turn1_query},
        timeout=120,
    )
    if r1.status_code != 200:
        print(f"❌ Turn 1 failed: {r1.status_code}")
        print(r1.text)
        sys.exit(1)

    data1 = r1.json()
    pretty_print_response("TURN 1 RESPONSE", data1)
    session_id = data1["session_id"]

    # ── TURN 2 (with the same session_id) ─────────────────────────
    turn2_query = "aur agar woh court mein nahi aaye to kya hota hai?"
    print(f"\n→ Sending turn 2 (with session_id {session_id[:8]}...): {turn2_query!r}")

    r2 = requests.post(
        f"{BASE_URL}/ask",
        headers={"Content-Type": "application/json"},
        json={"query": turn2_query, "session_id": session_id},
        timeout=120,
    )
    if r2.status_code != 200:
        print(f"❌ Turn 2 failed: {r2.status_code}")
        print(r2.text)
        sys.exit(1)

    data2 = r2.json()
    pretty_print_response("TURN 2 RESPONSE", data2)

    # ── Quality check ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  QUALITY CHECK")
    print('='*70)
    answer2 = data2.get("answer", "").lower()

    deposit_keywords = ["deposit", "landlord", "rent controller", "tenancy", "ex parte", "ek tarfa", "lahore", "punjab"]
    hits = [kw for kw in deposit_keywords if kw in answer2]
    print(f"Deposit-context keywords found in turn 2 answer: {hits}")
    if hits:
        print("✅ PASS — turn 2 references the deposit/landlord context from turn 1")
    else:
        print("⚠️  WARN — turn 2 answer may not be using turn 1 context. Read the answer manually to confirm.")


if __name__ == "__main__":
    main()