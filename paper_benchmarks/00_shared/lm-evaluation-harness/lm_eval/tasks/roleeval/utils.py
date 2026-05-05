import pandas as pd


def doc_to_text(doc):
    """Convert a document to text prompt for role-eval."""
    question = doc["question"]
    option_a = doc["A"]
    option_b = doc["B"]
    option_c = doc["C"]
    option_d = doc["D"]
    return f"{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:"


def doc_to_choice(doc):
    """Return the list of choices."""
    return ["A", "B", "C", "D"]
