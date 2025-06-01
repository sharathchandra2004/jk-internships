from nltk.data import find

try:
    find('tokenizers/punkt')
    find('tokenizers/punkt_tab')
    print("NLTK resources are available.")
except LookupError as e:
    print(f"Missing NLTK resource: {e}")
