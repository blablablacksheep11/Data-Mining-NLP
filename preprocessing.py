import pandas as pd
import re
import html
import emoji
import nltk

from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str, remove_numbers: bool = False,
               remove_repeated_punctuations: bool = True):

    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Decode HTML entities
    text = html.unescape(text)

    # 3. Convert emojis into textual descriptions
    text = emoji.replace_emoji(
        text,
        replace=lambda chars, data: " " + emoji.demojize(chars) + " "
    )

    # 4. Convert Unicode characters into ASCII equivalents
    text = unidecode(text)

    # 5. Remove URLs
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)

    # 6. Remove usernames and mentions
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_]+", " ", text)

    # 7. Remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)

    # 8. Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # 9. Remove decorative symbols while preserving emoji descriptions
    text = re.sub(r"[^\w\s.,!?:]", "", text)

    # 10. Handle numbers
    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    # 11. Reduce repeated characters
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # 12. Reduce repeated punctuation
    if remove_repeated_punctuations:
        text = re.sub(r"[!?.,]{3,}", lambda match: match.group(0)[0], text)

    # 13. Remove specific meaningless number
    text = re.sub(r"\b1635465\b", "", text)

    # 14. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 15. Convert to lowercase
    text = text.lower()

    return text


def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ

    elif tag.startswith('V'):
        return wordnet.VERB

    elif tag.startswith('N'):
        return wordnet.NOUN

    elif tag.startswith('R'):
        return wordnet.ADV

    else:
        return wordnet.NOUN


def preprocess_text(text):

    # 1. Tokenization
    tokens = word_tokenize(text)

    # 2. Stopword removal
    tokens = [
        word for word in tokens
        if word.lower() not in stop_words
    ]

    # 3. POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # 4. Lemmatization
    lemmatized_words = []

    for word, tag in pos_tags:

        pos = get_wordnet_pos(tag)

        lemma = lemmatizer.lemmatize(word, pos)

        lemmatized_words.append(lemma)

    return " ".join(lemmatized_words)


def prepare_message(text):

    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)

    return processed
