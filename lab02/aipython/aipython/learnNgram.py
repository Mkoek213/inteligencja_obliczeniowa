# learnNgram.py - Bigram and N-gram with embeddings
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
from display import Displayable
from learnTokenizer import Multiset, Tokenizer

class Bigram(Displayable):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ttc = [[0 for _ in tokenizer.tokens] for _ in tokenizer.tokens] # dense
        self.counts=[0 for _ in tokenizer.tokens]
        # ttc[token1][token2] is the number of times token 1 is followed by token 2
        generate_tokens = tokenizer.corpus_to_tokens()
        prev = next(generate_tokens)
        self.counts[prev] = 1
        for token in generate_tokens:
            self.ttc[prev][token] += 1
            self.counts[token] += 1
            prev = token
                                                                                    
    def generate(self, prompt, length, temp=1):
        res = prompt+""  # copy prompt
        ti = list(self.tokenizer.tokenize(prompt))[-1] # last token
        for i in range(length):
            ti = sample_i(self.ttc[ti], temp=temp)
            res += self.tokenizer.tokens[ti]
        return res

    def dist(self, prompt):
        """returns a sorted list of prob,string
        """
        ti = list(self.tokenizer.tokenize(prompt))[-1] # last token
        unnorm = self.ttc[ti]
        tot = sum(unnorm)
        return sorted([(n/tot,self.tokenizer.tokens[i]) for (i,n) in enumerate(unnorm) if n>0])
        

def sample_i(probs, temp=1):
    """Sample from an unnormalized probability distribution.
    probs is a list of nonnegative numbers
    """

    #keys = range(len(probs)) if isinstance(probs,list) else probs.keys()
    if isinstance(probs,list):
        probs = {i:v for (i,v) in  enumerate(probs)}
    if temp != 1:
       probs = {k:p**(1/temp) for k,p in probs.items()}
    tot = sum(probs.values())
    sm = 0
    p = random.random()*tot
    for i in probs:
       sm += probs[i]
       if p <= sm:
           return i
    assert False, f"sample_i has error with {probs=}"

# tok = Tokenizer(load_from_file ="tokens.pkl") #"tokens_wd_1000.pkl") #"tokens.pkl")
# bg = Bigram(tok)
# bg.generate(" this", 30, temp=0.2)
# bg.generate(" she", 30, temp=0.2)

# To get the distribution of the top next words given a text
# bg.dist(" this")[-20:]

# To determine how many instances of each token there are:
# sorted(list(zip(bg.counts,tok.tokens)))
# To determine probability of most likely tokens:
# sorted(list(zip([c/sum(bg.counts) for c in bg.counts],tok.tokens)))[-30:]

from learnNN import NN, Layer, Linear_complete_layer, Batch_generator, RMS_Prop
from learnProblem import Data_set

class One_Hot_Layer(Linear_complete_layer):
    """A complete linear layer where one input is 1 and the others are 0.
    No need to multiple by 1 or 0.
     """
    def __init__(self, nn, num_inputs, num_outputs, limit=None, final_layer=False):
          Linear_complete_layer.__init__(self, nn, num_outputs, limit, final_layer,
                                             num_inputs=num_inputs)

    def output_values(self, input, training=False):
        """Returns the outputs for the input values.
        It remembers the input for the backprop.
        """
        self.display(3,f"One-hot layer inputs: {input}")
        self.input=input
        for out in range(self.num_outputs):
            self.outputs[out] = (self.weights[input][out]
                                + self.weights[self.num_inputs][out])
        self.display(3,f"One-hot layer {input=}, {self.outputs=}")
        return self.outputs
          
    def backprop(self, errors):
        """Backpropagate errors, update weights, return input error.
        errors is a list of size self.num_outputs
        No need to return errors for layer's input as input is always data
        """
        self.display(3,f"One-hot Backprop. input: {self.input} output errors: {errors}")
        for out in range(self.num_outputs):
            inp=self.input
            #self.input_errors[inp] = self.weights[inp][out] * errors[out]
            self.delta[inp][out] += self.input * errors[out]
            self.delta[self.num_inputs][out] += errors[out]
        #self.display(3,f"One-hot layer backprop input errors: {self.input_errors}")
        #return self.input_errors

class Concat_Layers(Layer):
    """A layer created by concatenating a list of layers.
    Each layer has a single input value; 
          the input of the concatenation is list of these values
    Output of the concatenation is the concatenation of the outputs
    """
    def __init__(self, nn, layers):
        num_outputs = sum(lay.num_outputs for lay in layers)
        self.display(3, f"Concat_Layers {layers=} {num_outputs=}")
        Layer.__init__(self, nn, num_outputs)
        self.layers = layers

    def output_values(self, input_values, training=False):
        ov = []
        for lay,inp in zip(self.layers, input_values):
            ov += lay.output_values(inp)
        return ov

    def backprop(self, out_errors):
        # need to uppack the array of outputs to the output of each layer
        res=[]
        for lay in self.layers:
           res.append(lay.backprop(out_errors[:lay.num_outputs]))
           out_errors = out_errors[lay.num_outputs:]
        return res

class N_gram_dataset(Data_set):
    """creates dataset from slices of the token stream of length n
    """
    def __init__(self, n, tokenizer):
        self.n = n
        self.tokenizer = tokenizer
        self.input_features = [lambda e, iv=i: e[iv] for i in range(n-1)]
        self.target = lambda e:e[n-1]
        self.target.ftype = "categorical"
        self.target.frange = list(range(len(tokenizer.tokens)))
        self.train = []  # don't print anything for training set
        self.valid = []

class NGram_Batch_Generator(Batch_generator):
    def __init__(self, tokenizer, n):
        self.tokenizer = tokenizer
        self.n = n
        self.ngrams = self.Ngram_generator() # data generator

    def Ngram_generator(self):
        while True:  # keep going through the corpus
            token_gen = self.tokenizer.corpus_to_tokens()
            self.context = [next(token_gen) for i in range(self.n)]
            for token in token_gen:
                yield self.context
                self.context = self.context[1:]+[token]

    def get_batch(self, batch_size):
        return [next(self.ngrams) for i in range(batch_size)]

#Test:
# b = NGram_Batch_Generator(tok,5)
# b.get_batch(10)
    
class N_gram(NN):
    def __init__(self, n, tokenizer, hidden_size, optimizer=None, **hyperparms):
        self.n = n
        self.tokenizer = tokenizer
        dataset = N_gram_dataset(n, tokenizer)
        NN.__init__(self, dataset, batch_gen=NGram_Batch_Generator(tokenizer, n),
                         optimizer=optimizer, **hyperparms)
        self.add_layer(
            Concat_Layers(self,
                    [One_Hot_Layer(self, len(dataset.tokenizer.tokens),hidden_size) for i in range(n-1)]))
        # add hidden layers here
        self.add_layer(Linear_complete_layer(self, len(tokenizer.tokens)))

    def generate(self, prompt, length, temp=1):
        res = prompt+""  # copy prompt (to include prompt in result)
        toks = list(self.tokenizer.tokenize(prompt))
        if len(toks) >= self.n-1:
           context = toks[-self.n+1:]
        else:
           blank_token = self.tokenizer.token_trie[' '][0]
           context = [blank_token]*(self.n-1-len(prompt)) + toks
        for i in range(length):
            ti = sample_i(self.predictor(context), temp=temp)
            self.display(3,f"generate: {context=} {ti=} {self.tokenizer.tokens[ti]=}")
            res += self.tokenizer.tokens[ti]
            context = context[1:]+[ti]
        return res

# tok = Tokenizer()
# tok.create_tokens(200)
# fiveg = N_gram(5, tok, 20, optimizer= RMS_Prop)
# fiveg.learn(batch_size=100, num_iter = 10000, report_each=10000000)
# import time
# st=time.perf_counter(); fiveg.learn(batch_size=100, num_iter = 10000, report_each=10000000); et=time.perf_counter()
# fiveg.generate("the cat smiled and said ", 20)

# show distribution:
# sorted([(p,tok.tokens[t]) for (t,p) in fiveg.predictor(list(tok.tokenize("the cat smiled and said"))[-4:]).items()])[-20:]
# sorted([(p,tok.tokens[t]) for (t,p) in fiveg.predictor(list(tok.tokenize("the cat smiled and said "))[-4:]).items()])[-20:]  #space after s
# what is the context:
# tok.decode(list(tok.tokenize("the cat smiled and said"))[-4:])
# tok.decode(list(tok.tokenize("and said")))

