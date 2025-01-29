from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from spacy.matcher import Matcher
from num2words import num2words
import re

def solve(s):
    d = dict()
    for c in s:
        if c not in d:
            d[c] = 0
        d[c] += 1
    return ''.join(d.keys())

def init_configure(nlp):
    matcher = Matcher(nlp.vocab)
    
    pattern = [{"IS_DIGIT": True}]
    matcher.add("NUMBERS", [pattern])
    
    pattern = [{"TEXT": {"REGEX":r"@\w+"}}]
    matcher.add("TWITTER_MENTIONS", [pattern])
    
    pattern = [{'ORTH': '#'},{"TEXT": {"REGEX":r"\w+"}}]
    matcher.add("HASHTAGS", [pattern])
    
    tokenizer = Tokenizer(oov_token="<OOV>")
    
    max_length_tweet = 21*5
    max_length_keyword = 3*2
    return matcher, tokenizer, max_length_tweet, max_length_keyword

def get_list(text, c='#'):
    regex = ""
    if c == "https":
        regex = r"("+c+"?://[^\s]+)"
    elif c == "punct":
        regex = r"[!$%&´\'\"\(\)*,-./:;<=>?\\^_`\[\]{|}~+]"
    else:
        regex = r"("+c+"[^\s]+)"
    return re.findall(regex, text)

def clear_tweet(text, list_c):
    for c in list_c:
        if c in ["#", "@"]:
            lista = c
        else:
            lista = get_list(text, c)
        for item in lista:
            text = text.replace(item, " ").replace("  ", " ")
    text = re.sub(r'[^\x00-\x7F]+','', text)
    return text

def cleaner(text):
    hashtags = get_list(text, c='#')
    mentions = get_list(text, c='@')
    urls = get_list(text, c='https')
    clean_text = clear_tweet(text, ["#", "@", "https"])
    punctuations = get_list(clean_text, c="punct")
    clean_text = clear_tweet(clean_text, ["punct"])
    
    return clean_text, len(clean_text.split()), len(clean_text), len(hashtags), len(mentions), len(urls), len(punctuations)

def lemmatize(text, nlp):
    doc = nlp(text)
    for token in doc:
        text = text.replace(token.text, token.lemma_)
    return text

def remove_stopwords(text, nlp, dict_words):
    doc = nlp(text)
    filtered_sentence = []
    for token in doc:
        if (token.is_stop == False
            and len(token) > 2
            and token.is_ascii
            and token.is_alpha
            and token.text in dict_words[token.text[0]][token.text[1]]):
            filtered_sentence.append(token.text)
                
    return ' '.join(filtered_sentence)

def preprocessing_text(text, nlp, matcher, dict_words):
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        if doc.vocab.strings[match_id] == "NUMBERS":
            text = text.replace(doc[start:end].text, num2words(doc[start:end].text))
            
    text, n_words, n_characters, n_hashtags, n_mentions, n_urls, n_punctuations = cleaner(text)
    text = lemmatize(text, nlp)
    text = text.lower()
    clean_text = remove_stopwords(text, nlp, dict_words)
        
    return clean_text, n_words, n_characters, n_hashtags, n_mentions, n_urls, n_punctuations

def replace_spaces(text):
    text = text.replace("%20", " ")
    return text

def preprocessing_keyword(text, nlp):
    text = replace_spaces(text)
    clean_text = lemmatize(text, nlp)
    return clean_text

def tokenize(text, tokenizer, fit=False):
    if fit == True:
        tokenizer.fit_on_texts(text)
    return tokenizer.texts_to_sequences(text)

def get_tokens(text=None, max_length=None, tokenizer=None, fit=False, padding=False):
    preprocessed_text = tokenize(text, tokenizer, fit)
    if padding:
        preprocessed_text = pad_sequences(preprocessed_text, maxlen=max_length, padding='post')
    return preprocessed_text