from lib2to3.pgen2 import token
import re
from vncorenlp import VnCoreNLP
from underthesea import word_tokenize, sent_tokenize

def _clean_sentences(string):
    strip_special_chars = re.compile("[^\w0-9 ]+")
    string = re.sub(r"[\.,\?]+$-", "", string)
    string = string.lower().replace("<br />", " ")
    string = string.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    return re.sub(strip_special_chars, "", string.lower())

def _vn_tokenize(string : str) -> str:

    return word_tokenize(string, format = "text")


tokens = _vn_tokenize("Xin chào mọi người, mọi ăn ở đây rất là ngon. Vui chơi giải trí cũng tuyệt vời!")
print(tokens)