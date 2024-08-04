import re

# Most common abbrevations in corpus and other small things to substitute
ABBREVATIONS = {
    "e.g.": "for example",
    "E.g.": "for example",
    "U.S.": "united states",
    "w.r.t.": "with respect to",
    "i.e.": "that is",
    "i.i.d.": "independent and identically distributed",
    "i.i.": "independent and identically",
    "v.s.": "versus", "vs.": "versus",
    "etc.": "and so on", #TODO: besser et cetera? oder ist das zu exotisch
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
    "e2e": "end-to-end",
    "E2E": "end-to-end",
    "iii)": "", "ii)": "", "i)": "", "iv)": "", "v)": "",
    "?": ".", "!": ".",
    "a)": "", "b)": "", "c)": "", "d)": "", "e)": ""
}

# Most common letter-number-combinations that will not be substituted
letter_number_exceptions = ["L2","F1","L1","F2","seq2seq","Seq2Seq","word2vec","Word2Vec","2D"]

def preprocess_text(text, lower=False, years=False, percentages=False, numbers=False, 
                        letter_numbers=False, abbrev=False, special_characters=False):
    if lower:
        text = text.lower()

    # Substitue Years
    if years:
        text = re.sub(r'\b(19|20)\d{2}\b', 'YEAR', text)

    # Substitue Percentages
    if percentages:
        text = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%', "PERCENTAGE", text)

    # Substitute all remaining numbers
    if numbers:
        text = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', 'NUMBER', text)

    # Remove words that are combinations of letters and numbers 
    # (except L2, F1, word2vec, ... common in corpus and probably important for context)
    if letter_numbers:
        #pattern = r'\b(?!(L2|F1|L1|F2|seq2seq|word2vec|Seq2Seq|Word2Vec|2D)\b)\w*\d+\w*\b'
        pattern = rf'\b(?!({"|".join(letter_number_exceptions)})\b)\w*\d+\w*\b'
        text = re.sub(pattern, '', text)

    # Substitute most common abbrevations
    if abbrev:
        for abbrevation, substitution in ABBREVATIONS.items():
            text = text.replace(abbrevation, substitution)

    # Remove all characters that are not normal text
    if special_characters:
        text = re.sub(r'[^a-zA-Z0-9\s\-\.\,]', '', text)

    # Punkt hinter Titel des papers setzen, falls bert genutzt wird, [SEP] token hinter titel setzten.?????
    if len(text.split("\n\n")) < 2:
        text += "."
    else:
        text = re.sub(r'\n\n', ". ", text)

    # Aufeinanderfolgende whitespaces durch einzelnes blank ersetzen.
    text = re.sub(r'\s+', ' ', text).strip()

    return text