# learnTokenizer.py - Language Tokenizer
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
import pickle  # for saving tokenizations
from display import Displayable

train_corpus = ["pg11.txt","pg45.txt", "pg74.txt", "pg84.txt","pg1342.txt", "pg1661.txt"]
train_corpus_folder = "corpus/"
"""Training corpus:
    pg11 Alice's Adventures in Wonderland              163916 chars
    pg45 Anne of Green Gables                          580415 chars
    pg74 The Adventures of Tom Sawyer, Complete        412054 chars
    pg84 Frankenstein; Or, The Modern Prometheus       438806 chars
    pg1342 Pride and Prejudice                         748126 chars
    pg1661 The Adventures of Sherlock Holmes           581565 chars
"""

def file_to_chars0(file_name):
    with open(file_name,'r') as file:
        while c := file.read(1):
            if c == "\n":
                yield " "
            else:
                yield c

def file_to_chars(file_name):
    f2ch = file_to_chars0(file_name)
    try:
        lookahead = next(f2ch)
    except StopIteration:       # for empty files
        return
    for ch in f2ch:
        if lookahead != ' ' or ch !=' ':  #reject spaces after space
            yield lookahead
            lookahead = ch
    yield lookahead
            
def sequence_generators(*gens):
    """
    Given generators as arguments, generates the values of the generators in turn
    """
    for g in gens:
        # yield from g   # does not work
        for e in g:
            yield e

class Multiset(object):
    def __init__(self):
        self.bag = {}

    def add(self, elt):
        if elt in self.bag:
            self.bag[elt] += 1
        else:
            self.bag[elt] = 1

    def items(self):
        return self.bag.items()
    
class Tokenizer(Displayable):
    def __init__(self, corpus = train_corpus,
                       corpus_folder = train_corpus_folder,
                    load_from_file = None):
        self.corpus = corpus
        self.corpus_folder = corpus_folder
        if load_from_file is not None:
            self.load(load_from_file)
        else:
            self.tokens = ["[UNK]", " ", ",", ".",";", "<start>","<end>"]  # this could also be populated with characters in order.
            self.token_trie = {c:[t,{}] for t,c in enumerate(self.tokens)} # char -> [tok, dict]   tok is a token or None. dict is a token_trie
            self.token_pairs = Multiset()  # multiset of pairs of tokens
            for file_name in corpus:
                previous, _  =  self.token_trie["<start>"] # start token
                for ch in file_to_chars(corpus_folder+file_name):
                    if ch not in self.token_trie:
                        token = len(self.tokens)
                        self.tokens.append(ch)
                        self.token_trie[ch]= [token, {}]
                    else:
                        token,_ = self.token_trie[ch]
                    if self.legal_token_pair(previous, token):
                        self.token_pairs.add((previous, token))
                    previous = token
                if self.legal_token_pair(previous, token):
                    self.token_pairs.add((previous, token)) 
            self.display(1, f"Corpus has {len(self.tokens)} single-character tokens.")

    def legal_token_pair(self, previous, token):
        """If there are restrictions on token pairs put them here. Otherwise
        return True
        the sets should probably be constants and only evaluated once.
         """
        #return True   # no restrictions
        
        # if token cannot be a space, it only allows spaces at start and middle of tokens:
        return ( token not in {self.token_trie[ch][0] for ch in " ,.;"})
        
        #If previous and token cannot both be " ", it only allows subword tokens:           
        #return (token not in {self.token_trie[ch][0] for ch in " ,.;"} 
        #         or previous not in {self.token_trie[ch][0]
        #                                 for ch in " ,.;"})
 
    def decode(self, token_list):
        res = ""
        for tok in token_list:
            res += self.tokens[tok]
        return res

    def corpus_to_tokens(self):
        for file_name in self.corpus:
            char_generator = file_to_chars(self.corpus_folder+file_name)
            yield from self.tokenize(char_generator)

    def tokenize(self, string_input):
        """generate tokens from a string.
        """
        gen_input = (c for c in string_input) 
        lookahead = []
        unused = []
        ended = False
        while not ended:
            dic = self.token_trie
            unusedlist = list(unused)
            self.display(3,f"start while {lookahead=} {unusedlist=}")
            unused = (e for e in lookahead + unusedlist)
            ended = True
            for next_char in sequence_generators(unused, gen_input):
                if next_char not in self.token_trie:
                    next_char = self.tokens[0] # unknown character
                ended = False
                self.display(3,f"{next_char=}")
                if next_char in dic:
                    tok,dic = dic[next_char]
                    if tok is not None:  #dictionary entry is a token
                        self.display(3,f"token={tok=} {self.tokens[tok]}")
                        token=tok
                        lookahead=[]
                    else:
                        lookahead.append(next_char)
                        self.display(3,f"tok=None  {token=} {self.tokens[token]} {lookahead=}")
                else:
                    lookahead.append(next_char)
                    self.display(3,f"in else yielding {token=} {self.tokens[token]} {lookahead=}")
                    yield token
                    break
            else:
               if not ended:
                   self.display(3,f"for ended yielding {token=} {self.tokens[token]} {lookahead=} {ended=}")
                   yield token

    def create_tokens(self, num_tokens):
        while len(self.tokens) < num_tokens:
            count, (t0,t1) = max((c,p) for (p,c) in self.token_pairs.items())
            new_token = len(self.tokens)
            token_string = self.tokens[t0]+self.tokens[t1] # append tokens
            self.display(0, f'{len(self.tokens)}: Creating token: {self.tokens[t0]!r}+{self.tokens[t1]!r}'
                            f' -> {token_string!r} {count=}')
            self.tokens.append(token_string) 
            self.add_to_trie(token_string, new_token)

            # Rebuild pairs from scratch
            self.token_pairs = Multiset()
            generate_tokens = self.corpus_to_tokens()
            prev = next(generate_tokens)
            for token in generate_tokens:
                if self.legal_token_pair(prev, token):
                    self.token_pairs.add((prev, token))
                prev = token

    def add_to_trie(self, token_string, new_token):
        """Adds a new token to the current trie
        token_string is the string tham kaes up the token
        new_token is the index into tokens list.
        """
        token_trie = self.token_trie
        for ch in token_string[:-1]:
            if ch in token_trie:
                (_, token_trie) = token_trie[ch]
            else:
                dct = {}
                token_trie[ch] = [None,dct]
                token_trie = dct
        if token_string[-1] in token_trie:
            token_trie[token_string[-1]][0] = new_token
        else:
            token_trie[token_string[-1]] = [new_token,{}]

    # Saving and loading computed data structures
    def save(self, filename="tokens.pkl"):
        file = open(filename, 'wb')
        pickle.dump((self.tokens,self.token_trie,self.token_pairs), file)
    def load(self, filename="tokens.pkl"):
        file = open(filename, 'rb')
        (self.tokens,self.token_trie,self.token_pairs) = pickle.load(file)
        self.display(0, f"Corpus has {len(self.tokens)} tokens.")

if __name__ == "__main__":
   print("""# Try:
tok = Tokenizer()
tok.create_tokens(200)
# OR
tok = Tokenizer(load_from_file ="tokens_wd_1000.pkl") #"tokens.pkl")
#tok.load()
""")

# 10 most likely pairs:
# sorted((n, (tok.tokens[t1], tok.tokens[t2])) for ((t1,t2),n) in tok.token_pairs.items())[-10:]
# what token will be created? create a new token: tok.create_tokens(201)
# how have the 10 most likely pairs changed?

# longest tokens:
# sorted([(len(e),e) for e in tok.tokens])[-20:]

