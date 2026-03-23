# learnLinear.py - Linear Regression and Classification
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from learnProblem import Learner
import random, math

class Linear_learner(Learner):
    def __init__(self, dataset, train=None, 
                 learning_rate=0.01, max_init = 0, squashed=True, bias0=None):
        """Creates a gradient descent searcher for a linear classifier.
        The main learning is carried out by learn()
        
        dataset provides the target and the input features
        train provides a subset of the training data to use
        learning_rate is the gradient descent step size
        max_init is the maximum absolute value of the initial weights (does this matter?)
        squashed specifies whether the output is a squashed linear function
        bias0 is initial bias if not None
        """
        self.dataset = dataset
        self.target = dataset.target
        if train==None:
            self.train = self.dataset.train
        else:
            self.train = train
        self.learning_rate = learning_rate
        self.squashed = squashed
        self.input_features = [one]+dataset.input_features # one is defined below
        self.weights = {feat:random.uniform(-max_init,max_init)
                        for feat in self.input_features}
        if bias0 is not None:
            self.weights[one] = bias0
        

    def predictor(self,e):
        """returns the prediction of the learner on example e"""
        linpred = sum(w*f(e) for f,w in self.weights.items())
        if self.squashed:
            return sigmoid(linpred)
        else:
            return linpred

    def __str__(self, sig_dig=3):
        """returns the doc string for the current prediction function
        sig_dig is the number of significant digits in the numbers"""
        doc = "+".join(str(round(val,sig_dig))+"*"+feat.__doc__
                        for feat,val in self.weights.items())
        if self.squashed:
            return "sigmoid("+ doc+")" 
        else:
            return doc

    def learn(self, batch_size=32, num_iter=1000):
        batch_size = min(batch_size, len(self.train))
        d = {feat:0 for feat in self.weights}
        for it in range(num_iter):
            self.display(2,f"prediction= {self}")
            for e in random.sample(self.train, batch_size):
                error =  self.predictor(e) - self.target(e)
                for feat in self.weights:
                    d[feat] +=  error*feat(e)
            for feat in self.weights:
                self.weights[feat] -=  self.learning_rate*d[feat]
                d[feat]=0
        return self.predictor

def one(e):
    "1"
    return 1

def sigmoid(x):
    return 1/(1+math.exp(-x))

def logit(x):
    return -math.log(1/x-1)

def softmax(xs, domain=None):
    """xs is a list of values, and 
    domain is the domain (a list) or None if the list should be returned
    returns a distribution over the domain (a dict)
    """
    m = max(xs)  # use of m prevents overflow (and all values underflowing)
    exps = [math.exp(x-m) for x in xs]
    s = sum(exps)
    if domain:
        return {d:v/s for (d,v) in zip(domain,exps)}
    else:
        return [v/s for v in exps]

def indicator(v, domain):
    return [1 if v==dv else 0 for dv in domain]
    
from learnProblem import Data_set, Data_from_file, Evaluate
from learnProblem import Evaluate
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = Data_from_file('data/SPECT.csv', target_index=0)
    # data = Data_from_file('data/mail_reading.csv', target_index=-1)
    # data = Data_from_file('data/carbool.csv', one_hot=True, target_index=-1)
    ll = Linear_learner(data)
    ll.learn()
    ll.evaluate()

def plot_steps(data,
               learner=None,
               criterion=Evaluate.squared_loss,
               step=1,
               num_steps=1000,
               log_scale=True,
               legend_label=""):
    """
    plots the training and validation error for a learner.
    data is the dataset
    learner is the learning algorithm (default is linear learner on the data)
    criterion gives the evaluation criterion plotted on the y-axis
    step specifies how many steps are run for each point on the plot
    num_steps is the number of points to plot
    
    """
    if legend_label != "": legend_label+=" "
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("step")
    ax.set_ylabel("Average "+criterion.__doc__)
    if log_scale:
        ax.set_xscale('log')  #plt.semilogx()  #Makes a log scale
    else:
        ax.set_xscale('linear')
    if learner is None:
        learner = Linear_learner(data)
    train_errors = []
    valid_errors = []
    for i in range(1,num_steps+1,step):
        valid_errors.append(data.evaluate_dataset(data.valid, learner.predictor, criterion))
        train_errors.append(data.evaluate_dataset(data.train, learner.predictor, criterion))
        learner.display(2, "Train error:",train_errors[-1],
                          "Valid error:",valid_errors[-1])
        learner.learn(num_iter=step)
    ax.plot(range(1,num_steps+1,step),train_errors,ls='-',label=legend_label+"training")
    ax.plot(range(1,num_steps+1,step),valid_errors,ls='--',label=legend_label+"validation")
    ax.legend()
    #plt.draw()
    learner.display(1, "Train error:",train_errors[-1],
                          "Validation error:",valid_errors[-1])

# This generates the figure
# from learnProblem import Data_set_augmented, prod_feat
# data = Data_from_file('data/SPECT.csv', prob_valid=0.5, target_index=0, seed=123)
# dataplus = Data_set_augmented(data, [], [prod_feat])
# plot_steps(data, num_steps=1000)
# plot_steps(dataplus, num_steps=1000)  # warning slow

def arange(start,stop,step):
    """enumerates values in the range [start,stop) separated by step.
    like range(start,stop,step) but allows for integers and floats.
    Rounding errors are expected with real numbers. (or use numpy.arange)
    """
    while start<stop:
        yield start
        start += step

def plot_prediction(data,
               learner = None,
               minx = 0,
               maxx = 5,
               step_size = 0.01,    # for plotting
               label = "function"):
    plt.ion()
    fig,ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if learner is None:
        learner = Linear_learner(data, squashed=False)
    learner.learning_rate=0.001
    learner.learn(num_iter=100)
    learner.learning_rate=0.0001
    learner.learn(num_iter=1000)
    learner.learning_rate=0.00001
    learner.learn(num_iter=10000)
    learner.display(1,f"function learned is {learner}. "
              "error=",data.evaluate_dataset(data.train, learner.predictor, Evaluate.squared_loss))
    ax.plot([e[0] for e in data.train],[e[-1] for e in data.train],"bo",label="data")
    ax.plot(list(arange(minx,maxx,step_size)),
             [learner.predictor([x])
                for x in arange(minx,maxx,step_size)],
             label=label)
    ax.legend(loc='upper left')
    
from learnProblem import Data_set_augmented, power_feat
def plot_polynomials(data,
                learner_class = Linear_learner,
                max_degree = 5,
                minx = 0,
                maxx = 5,
                num_iter = 1000000,
                learning_rate = 0.00001,
                step_size = 0.01,   # for plotting
                ):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot([e[0] for e in data.train],[e[-1] for e in data.train],"ko",label="data")
    x_values = list(arange(minx,maxx,step_size))
    line_styles = ['-','--','-.',':']
    colors = ['0.5','k','k','k','k']
    for degree in range(max_degree):
        data_aug = Data_set_augmented(data,[power_feat(n) for n in range(1,degree+1)],
                                          include_orig=False)
        learner = learner_class(data_aug,squashed=False)
        learner.learning_rate = learning_rate
        learner.learn(num_iter=num_iter)
        learner.display(1,f"For degree {degree}, "
                     f"function learned is {learner}. "
                     "error=",data.evaluate_dataset(data.train, learner.predictor, Evaluate.squared_loss)) 
        ls = line_styles[degree % len(line_styles)]
        col = colors[degree % len(colors)]
        ax.plot(x_values,[learner.predictor([x]) for x in x_values], linestyle=ls, color=col,
                          label="degree="+str(degree))
        ax.legend(loc='upper left')

# Try:
# data0 = Data_from_file('data/simp_regr.csv', prob_test=0, prob_valid=0, one_hot=False, target_index=-1)
# plot_prediction(data0)
# Alternatively:
# plot_polynomials(data0)
# What if the step size was bigger?
#datam = Data_from_file('data/mail_reading.csv', target_index=-1)
#plot_prediction(datam)

