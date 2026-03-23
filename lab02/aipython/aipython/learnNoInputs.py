# learnNoInputs.py - Learning ignoring all input features
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from learnProblem import Evaluate
import math, random, collections, statistics
import utilities  # argmax for (element,value) pairs

class Predict(object):
    """The class of prediction methods for a list of values.
    The doc strings the same length because they are used in tables.
    Note that the methods don't have the self argument.
    To use call Predict.laplace(data) etc."""

    ### The following return a distribution over values (for classification)
    def empirical(data, domain=[0,1], icount=0):
        "empirical dist   "
        # returns a distribution over values
        # icount is pseudo count for each value
        counts = {v:icount for v in domain}
        for e in data:
            counts[e] += 1
        s = sum(counts.values())
        return {k:v/s for (k,v) in counts.items()}

    def laplace(data, domain=[0,1]):
        "Laplace         "  # for categorical data
        return Predict.empirical(data, domain, icount=1)

    def cmode(data, domain=[0,1]):
        "mode            " # for categorical data
        md = statistics.mode(data)
        return {v: 1 if v==md else 0 for v in domain}

    def cmedian(data, domain=[0,1]):
        "median          " # for categorical data
        md = statistics.median_low(data)  # always return one of the values
        return {v: 1 if v==md else 0 for v in domain}
        
    ### The following return a single prediction (for regression).
    ### The domain argument is ignored.
    
    def mean(data, domain=[0,1]):
        "mean            "
        # returns a real number
        return statistics.mean(data)

    def rmean(data, domain=[0,1], mean0=0, pseudo_count=1):
        "regularized mean"
        # returns a real number.
        # mean0 is the mean to be used for 0 data points
        # With mean0=0.5, pseudo_count=2, same as laplace for [0,1] data
        sm = mean0 * pseudo_count
        count = pseudo_count
        for e in data:
            sm += e
            count += 1
        return sm/count

    def mode(data, domain=[0,1]):
        "mode            "
        return statistics.mode(data)

    def median(data, domain=[0,1]):  
        "median          "
        return statistics.median(data)

    all = [empirical, mean, rmean, laplace, cmode, mode, median, cmedian]

    # The following suggests appropriate predictions as a function of the target type
    select = {"boolean": [empirical, laplace, cmode, cmedian],
              "categorical": [empirical, laplace, cmode, cmedian],
              "numeric": [mean, rmean, mode, median]}

from learnProblem import Learner #, Data_set

class Naive_learner(Learner):
    def __init__(self, dataset, predictor=Predict.mean, **predictor_args):
        """returns a predictor that makes the same prediction for every example"""
        self.dataset = dataset
        self.predicted_value = predictor([dataset.target(e) for e in dataset.train], **predictor_args)
        def pred(e):
            return self.predicted_value
        pred.__doc__ = f"predict {self.predicted_value} for each value"
        self.predictor = pred
        
    def learn(self):
        return self.predictor

    def __str__(self):
       return f"predict {self.predicted_value} for each value"

def test_no_inputs(error_measures = Evaluate.all_criteria, num_samples=10000,
                       test_size=10,  training_sizes= [1,2,3,4,5,10,20,100,1000]):
    for train_size in training_sizes:
        results = {predictor: {error_measure: 0 for error_measure in error_measures}
                        for predictor in Predict.all}
        for sample in range(num_samples):
             prob = random.random()
             training = [1 if random.random()<prob else 0 for i in range(train_size)]
             test = [1 if random.random()<prob else 0 for i in range(test_size)]
             for predictor in Predict.all:
                 prediction = predictor(training)
                 for error_measure in error_measures:
                     results[predictor][error_measure] += sum( error_measure(prediction,actual)
                                                                   for actual in test) / test_size
        print(f"For training size {train_size}:")
        print("   Predictor\t","\t".join(error_measure.__doc__ for
                                           error_measure in error_measures),sep="\t")
        for predictor in Predict.all:
            print(f"   {predictor.__doc__}",
                      "\t".join("{:.7f}".format(results[predictor][error_measure]/num_samples)
                                    for error_measure in error_measures),sep="\t")
        
if __name__ == "__main__":
    test_no_inputs()
        
