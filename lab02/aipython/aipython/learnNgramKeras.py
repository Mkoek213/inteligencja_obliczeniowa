# learnNgramKeras.py - Bigram and N-gram with embeddings (using Keras)
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

### learn n-gram with embeddings using Keras

import keras
import tensorflow as tf
import numpy as np
import random
from learnTokenizer import Tokenizer, Multiset, file_to_chars
import matplotlib.pyplot as plt

context_size = 20
embedding_size = 500
                 
# load tokenizer that is already saved:
tok = Tokenizer(load_from_file ="tokens_wd_1000.pkl") #"tokens.pkl")
#otherwise:
#tok = Tokenizer()
# tok.create_tokens(200)

# Get these books and put them in folder corpus_test/
# https://www.gutenberg.org/cache/epub/12/pg12.txt  Through the Looking-Glass
# https://www.gutenberg.org/cache/epub/76/pg76.txt  Adventures of Huckleberry Finn
# https://www.gutenberg.org/cache/epub/2701/pg2701.txt Moby Dick
# https://www.gutenberg.org/cache/epub/35461/pg35461.txt A Short History of the World
# which are similar to training books?

test_corpus = ["pg12","pg76", "pg2701", "pg35461"]
test_corpus_folder = "corpus_test/"
test_seqs = [np.array(list(tok.tokenize(file_to_chars(test_corpus_folder+fn+".txt"))))
                 for fn in test_corpus]

token_seq = np.array(list(tok.corpus_to_tokens()))
unigram_inputs = np.array([ [] for _ in tok.corpus_to_tokens()])

def dataset_from_file(tokenizer, filename, context_size=context_size):
    file_tokens = np.array(list(tokenizer.tokenize(file_to_chars(filename))))
    dataset = keras.utils.timeseries_dataset_from_array(
            data = file_tokens[:-context_size],
            targets = file_tokens[context_size:],
            sequence_length = context_size
            )
    return dataset

class Unigram(object):
    def __init__(self):
        inputs = keras.Input(shape=(0,), dtype="int32")
        embedding = keras.layers.Embedding(
            input_dim = 0,
            output_dim = len(tok.tokens)
            )
        output =  keras.layers.Softmax()(embedding(inputs))
        self.model = keras.Model(inputs, output, name=f"Unigram")
        self.model.compile(optimizer="adam", #"rmsprop",
              loss="sparse_categorical_crossentropy", 
              metrics=["sparse_categorical_accuracy"])
        self.model.summary()

#unigram = Unigram()
#unigram.model.fit(unigram_inputs, token_seq, epochs=2)

optimizer = "adam" #"rmsprop" #

class NGram(object):
    def __init__(self,
                 context_size = context_size,
                 embedding_size = embedding_size,
                 dataset = None
                 ):

        self.context_size = context_size
        self.embedding_size = embedding_size
        if dataset is None:
            self.dataset = keras.utils.timeseries_dataset_from_array(
                data = token_seq[:-self.context_size],
                targets = token_seq[self.context_size:],
                sequence_length = self.context_size,
                shuffle=True 
                )
        else:
            self.dataset = dataset
        inputs = keras.Input(shape=(self.context_size,), dtype="int32")
        embedding = keras.layers.Embedding(
            input_dim = len(tok.tokens),
            output_dim = self.embedding_size
            #,mask_zero=True
            )
        emb = embedding(inputs)
        con = keras.layers.Flatten()(emb)
        output = keras.layers.Dense(len(tok.tokens), activation="softmax")(con)
        self.model = keras.Model(inputs, output, name=f"NGram_{context_size}_{embedding_size}")

        self.model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",#keras.losses.SparseCategoricalCrossentropy(), #from_logits=False),
              metrics=["sparse_categorical_accuracy"])
        print(f"NGram({context_size=}, {embedding_size=})")
        self.model.summary()     # get statistics

    def generate(self, prompt, max_length=158):
        """generates a completion of prompt up to max_length tokens.
        filler is used so a prompt less than the context_size does not give an error
        """
        filler = [random.randint(0,len(tok.tokens)-1) for _ in range(self.context_size)]
        tokens = filler + list(tok.tokenize(prompt))
        #print(f"{tokens=}")
        prompt_length = len(tokens)
        for _ in range(max_length+context_size - prompt_length):
            prediction = self.model(keras.ops.convert_to_numpy([tokens[-self.context_size:]]))
            tokens.append(np.argmax(prediction).item())
        #print(f"{tokens=}")
        return "".join(tok.tokens[t] for t in tokens[self.context_size:])


#ngram = NGram(context_size = 20, embedding_size = 500)
#ngram.model.fit(ngram.dataset, epochs=100)

# generate("the monster said to Alice")
# 'the monster said to Alice, jumped up and down in an agony of terror. "Oh, there goes his _precious_ nose!" as an unusually large saucepan flew close by it, and'

# generate(" I dare not expect such success, Alice") # " I dare not expect such success, " from Frankenstein
# ' I dare not expect such success, Alice, in pictures, making it the trust you, and making them the very politeness to sitting for a moment, silence, she said, very stiffly, to Elizabeth,-- "I hope you are well, Miss Bennet'

# Testing:
# test1 =dataset_from_file(tok, 'corpus_test/pg12.txt')
# ngram.model.evaluate(test1)

def plot_errors(context_sizes = [4],
                embedding_sizes = [10],
                epochs = 10,
                repeat_epochs = 1  # must be > 0
                ):
    """
    exactly one of repeat_epochs, len(context_sizes), len(embedding_sizes) should be greater than one.
    That becomes the x-axis in the plot.
    The others must be 1.
    if repeat_epochs>1, normally epochs=1 (so it plots each epoch)
    """
    global fig, axs, train_results, test_results  # so they can be checked later
    error_name = ['loss', 'accuracy']
    train_results = []
    test_results = [[] for _ in test_corpus]
    plt.ion()
    fig, axs = plt.subplots(1,2)
    #ax.set_xscale('log')  # change between log and linear scale
    if repeat_epochs > 1:
        xlabel = "Epochs"
        xvalues = list(range(epochs,repeat_epochs*epochs+epochs,epochs))
        title = f"context_size={context_sizes[0]}, embedding_size={embedding_sizes[0]}, {optimizer=}"
    if len(context_sizes) > 1:
        xlabel = "Context size"
        xvalues = context_sizes
        title = f"epochs={epochs}, embedding_size={embedding_sizes[0]}, {optimizer=}"
    if len(embedding_sizes) > 1:
        xlabel = "Embedding size"
        xvalues = embedding_sizes
        title = f"epochs={epochs}, context_size={context_sizes[0]}, {optimizer=}"
    for err in range(2):
        axs[err].set_xlabel(xlabel)
        axs[err].set_ylabel(error_name[err])
    fig.suptitle(title)
    for context_size in context_sizes:
        dataset = keras.utils.timeseries_dataset_from_array(
            data = token_seq[:-context_size],
            targets = token_seq[context_size:],
            sequence_length = context_size,
            shuffle=True 
            )
        test_datasets =[keras.utils.timeseries_dataset_from_array(
            data = test_seq[:-context_size],
            targets = test_seq[context_size:],
            sequence_length = context_size,
            shuffle=True 
            ) for test_seq in test_seqs]
        for embedding_size in embedding_sizes:
            ngram = NGram(context_size = context_size,
                              embedding_size = embedding_size,
                              dataset=dataset)
            for ep in range(repeat_epochs):
                print(f"Training: ({context_size=}, {embedding_size=}, repeat# {ep+1})")
                hist = ngram.model.fit(dataset, epochs=epochs)
                train_results.append(ngram.model.evaluate(dataset))
                print("Test:")
                for i in range(len(test_corpus)):
                    test_results[i].append(ngram.model.evaluate(test_datasets[i]))
    for err in range(2):
        axs[err].plot(xvalues, [res[err] for res in train_results], label="training")
        for i in range(len(test_corpus)):
            axs[err].plot(xvalues, [res[err] for res in test_results[i]], label=test_corpus[i])
        axs[err].legend()
        
# plot_errors([10],[20],1,100)
# plot_errors([10],list(range(1,202,10)),10,1)
# plot_errors(list(range(1,52,5)),[20],10,1)

