"""
Adalat-AI Language Support Expansion Guide
==========================================
To add a new language to the router, follow these steps:

Supported languages currently:
- english
- roman_urdu
- german

Languages planned:
- punjabi
- sindhi
- pashto
- formal_urdu
- saraiki

Steps to add a new language:
1. Add detection keywords to LANGUAGE_KEYWORDS below
2. Update src/agents/router.py detect_language() prompt
3. Update src/agents/router.py translate_query() if needed
4. Test with sample queries
"""

LANGUAGE_KEYWORDS = {
    "roman_urdu": [
        "mera", "meri", "kya", "hai", "nahi", "wapas",
        "ghar", "rent", "deposit", "landlord", "paise",
        "court", "police", "arrest", "haq", "qanoon"
    ],
    "german": [
        "vermieter", "kaution", "miete", "nicht", "zurück",
        "gibt", "mein", "ist", "das", "der", "die", "wohnung"
    ],
    "punjabi": [
        "mera", "kina", "kithe", "hunda", "nahi",
        "paise", "makaan", "kiraya", "adalat"
    ],
    "sindhi": [
        "mون", "آهي", "ناهي", "گهر", "ڪرايو"
    ],
    "pashto": [
        "زما", "دی", "نه", "کور", "کرایه"
    ],
    "formal_urdu": [
        "میں", "ہے", "نہیں", "واپس", "مکان",
        "کرایہ", "ڈپازٹ", "عدالت", "قانون"
    ]
}

JURISDICTION_HINTS = {
    "roman_urdu": "PK",
    "punjabi": "PK",
    "sindhi": "PK",
    "pashto": "PK",
    "formal_urdu": "PK",
    "german": "DE",
    "english": None
}

if __name__ == "__main__":
    print("Language expansion config loaded.")
    print(f"Supported languages: {list(LANGUAGE_KEYWORDS.keys())}")