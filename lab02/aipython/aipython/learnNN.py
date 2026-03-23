# learnNN.py - Neural Network Learning
# AIFCA Python code Version 0.9.18 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2026 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from display import Displayable
from learnProblem import Learner, Data_set, Data_from_file, Data_from_files, Evaluate
from learnLinear import sigmoid, one, softmax, indicator
import random, math, time

class Layer(Displayable):
    def __init__(self, nn, num_inputs=None, num_outputs=None):
        """Abstract layer class, must be overridden.
        nn is the neural network this layer is part of
        num_outputs is the number of outputs for this layer.
        """
        self.nn = nn
        self.num_inputs = nn.num_outputs if num_inputs is None else num_inputs # nn output is layer's input 
        if num_outputs:
            self.num_outputs = num_outputs
        else:
            self.num_outputs =  self.num_inputs  # same as the inputs
        self.outputs= [0]*self.num_outputs
        self.input_errors = [0]*self.num_inputs
        self.weights = []

    def output_values(self, input_values, training=False):
        """Return the outputs for this layer for the given input values.
        input_values is a list (of length self.num_inputs) of the inputs
        returns a list of length self.num_outputs.
        It can act differently when training and when predicting.
        """
        raise NotImplementedError("output_values")    # abstract method

    def backprop(self, out_errors):
        """Backpropagate the errors on the outputs
        errors is a list of output errors (of length self.num_outputs).
        Returns list of input errors (of length self.num_inputs).
        
        This is only called after corresponding output_values(), 
           which should remember relevant information
        """
        raise NotImplementedError("backprop")    # abstract method

class Optimizer(Displayable):
    def update(self, layer):
        """updates parameters after a batch.
        """
        pass

class Linear_complete_layer(Layer):
    """a completely connected layer"""
    def __init__(self, nn, num_outputs, limit=None, final_layer=False, num_inputs=None):
        """A completely connected linear layer.
        nn is a neural network that the inputs come from
        num_outputs is the number of outputs
        the random initialization of parameters is in range [-limit,limit]
        """
        Layer.__init__(self, nn, num_inputs=num_inputs, num_outputs=num_outputs)
        if limit is None:
            limit =math.sqrt(6/(self.num_inputs+self.num_outputs))
        # self.weights[i][o] is the weight between input i and output o
        if final_layer:
             self.weights = [[0  if i < self.num_inputs
                                    or (nn.output_type != "categorical")
                               else 1
                            for o in range(self.num_outputs)]
                        for i in range(self.num_inputs+1)]
        else:
            self.weights = [[random.uniform(-limit, limit)
                            if i < self.num_inputs else 0
                            for o in range(self.num_outputs)]
                        for i in range(self.num_inputs+1)]
        # self.weights[i][o] is the accumulated change for a batch.
        self.delta = [[0 for o in range(self.num_outputs)]
                        for i in range(self.num_inputs+1)]

    def output_values(self, inputs, training=False):
        """Returns the outputs for the input values.
        It remembers the values for the backprop.
        """
        self.display(3,f"Linear layer inputs: {inputs}")
        self.inputs = inputs
        for out in range(self.num_outputs):
            self.outputs[out] = (sum(self.weights[inp][out]*self.inputs[inp]
                                       for inp in range(self.num_inputs))
                              + self.weights[self.num_inputs][out])
        self.display(3,f"Linear layer inputs: {inputs}")
        return self.outputs

    def backprop(self, errors):
        """Backpropagate errors, update weights, return input error.
        errors is a list of size self.num_outputs
        Returns errors  for layer's inputs of size 
        """
        self.display(3,f"Linear Backprop. input: {self.inputs} output errors: {errors}")
        for out in range(self.num_outputs):
            for inp in range(self.num_inputs):
                self.input_errors[inp] = self.weights[inp][out] * errors[out]
                self.delta[inp][out] += self.inputs[inp] * errors[out]
            self.delta[self.num_inputs][out] += errors[out]
        self.display(3,f"Linear layer backprop input errors: {self.input_errors}")
        return self.input_errors

class ReLU_layer(Layer):
    """Rectified linear unit (ReLU) f(z) = max(0, z).
    The number of outputs is equal to the number of inputs. 
    """
    def __init__(self, nn):
        Layer.__init__(self, nn)
        

    def output_values(self, input_values, training=False):
        """Returns the outputs for the input values.
        It remembers the input values for the backprop.
        """
        self.input_values = input_values
        for i in range(self.num_inputs):
            self.outputs[i] = max(0,input_values[i])
        return self.outputs

    def backprop(self,out_errors):
        """Returns the derivative of the errors"""
        for i in range(self.num_inputs):
            self.input_errors[i] = out_errors[i] if self.input_values[i]>0 else 0
        return self.input_errors

class Sigmoid_layer(Layer):
    """sigmoids of the inputs.
    The number of outputs is equal to the number of inputs. 
    Each output is the sigmoid of its corresponding input.
    """
    def __init__(self, nn):
        Layer.__init__(self, nn)

    def output_values(self, input_values, training=False):
        """Returns the outputs for the input values.
        It remembers the output values for the backprop.
        """
        for i in range(self.num_inputs):
            self.outputs[i] = sigmoid(out_errors[i])
        return self.outputs

    def backprop(self,errors):
        """Returns the derivative of the errors"""
        for i in range(self.num_inputs):
            self.input_errors[i] = input_values[i]*out_errors[i]*(1-out_errors[i])
        return self.input_errors

class NN(Learner):
    def __init__(self, dataset, batch_gen=None, optimizer=None, **hyperparms):
        """Creates a neural network for a dataset
        batch_gen is the algorithm used to generate batches (e.g., random, streaming)
        optimizer is the optimizer: default is SGD
        hyperparms is the dictionary of hyperparameters for the optimizer
        """
        self.batch_gen = Batch_generator(dataset.train) if batch_gen is None else batch_gen
        self.dataset = dataset
        self.optimizer = optimizer if optimizer else SGD
        self.hyperparms = hyperparms
        self.output_type = dataset.target.ftype
        self.input_features = dataset.input_features
        self.num_outputs = len(self.input_features) # empty NN
        self.layers = []
        self.bn = 0 # number of batches run
        self.printed_heading = False    # for tracing, so header printed once

    def add_layer(self, layer):
        """add a layer to the network.
        Each layer gets number of inputs from the previous layers outputs.
        """
        self.layers.append(layer)
        #if hasattr(layer, 'weights'):
        layer.optimizer = self.optimizer(layer, **self.hyperparms)
        self.num_outputs = layer.num_outputs

    def predictor(self,ex):
        """Predicts the value of the first output for example ex.
        """
        values = [f(ex) for f in self.input_features]
        for layer in self.layers:
            values = layer.output_values(values)
        return sigmoid(values[0]) if self.output_type =="boolean" \
               else softmax(values, self.dataset.target.frange) if self.output_type == "categorical" \
               else values[0]

    def learn(self, batch_size=32, num_iter = 100, report_each=10):
        """Learns parameters for a neural network using the chosen optimizer.
        batch_size is the size of each batch
        num_iter is the number of iterations over the batches
        report_each means print errors after each multiple of that number of batches
        """
        self.report_each = report_each
        if not self.printed_heading and num_iter >= report_each:
            self.display(1,"batch\tTraining\tTraining\tValidation\tValidation")
            self.display(1,"\tAcccuracy\tLog loss\tAcccuracy\tLog loss")
            self.printed_heading = True
            self.trace()
        for i in range(num_iter):
            batch = self.batch_gen.get_batch(batch_size) #random.sample(self.dataset.train, batch_size)
            for e in batch:
                # compute all outputs
                values = [f(e) for f in self.input_features]
                for layer in self.layers:
                    values = layer.output_values(values, training=True)
                # backpropagate
                predicted = [sigmoid(v) for v in values] \
                              if self.output_type == "boolean" \
                            else softmax(values) \
                              if self.output_type == "categorical" \
                            else values
                actuals = indicator(self.dataset.target(e), self.dataset.target.frange) \
                            if self.output_type == "categorical"\
                            else [self.dataset.target(e)]
                errors = [pred-obsd for (obsd,pred) in zip(actuals,predicted)]
                for layer in reversed(self.layers):
                    errors = layer.backprop(errors)
            # Update all parameters in batch
            for layer in self.layers:
                layer.optimizer.update(layer)
            self.bn+=1
            if (i+1)%report_each==0:
                self.trace()

    def trace(self):
        """print tracing of the batch updates"""
        self.display(1,self.bn,"\t",
            "\t\t".join("{:.4f}".format(
                    self.dataset.evaluate_dataset(data, self.predictor, criterion))
                            for data in [self.dataset.train, self.dataset.valid]
                            for criterion in [Evaluate.accuracy, Evaluate.log_loss]), sep="")

class Batch_generator(Displayable):
    """Generator of batches.
    Default implementation is to take a random subset of the dataset.
    This may not be applicable for streaming data.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def get_batch(self, batch_size):
        return random.sample(self.dataset, batch_size)
        
class SGD(Optimizer):
    """Vanilla SGD"""
    def __init__(self, layer, lr=0.01):
        """layer is a layer, which contains weight and gradient matrices
        Layers without weights have weights=[]
        """
        self.lr = lr
    
    def update(self, layer):
        """update weights of layer after a batch.
        """
        for inp in range(len(layer.weights)):
            for out in range(len(layer.weights[0])):  
                layer.weights[inp][out] -= self.lr*layer.delta[inp][out]
                layer.delta[inp][out] = 0

class Momentum(Optimizer):
    """SGD with momentum"""
    
    """a completely connected layer"""
    def __init__(self, layer, lr=0.01, momentum=0.9):
        """
        lr is the learning rate
        momentum is the momentum parameter 
        
        """
        self.lr = lr
        self.momentum = momentum
        layer.velocity = [[0 for _ in range((len(layer.weights[0])))]
                             for _ in range(len(layer.weights))]


    def update(self, layer):
        """updates parameters after a batch with momentum"""
        for inp in range(len(layer.weights)):
            for out in range(len(layer.weights[0])):
                layer.velocity[inp][out] = self.momentum*layer.velocity[inp][out] - self.lr*layer.delta[inp][out]
                layer.weights[inp][out] += layer.velocity[inp][out]
                layer.delta[inp][out] = 0
               
class RMS_Prop(Optimizer):
    """a completely connected layer"""
    def __init__(self, layer, rho=0.9, epsilon=1e-07, lr=0.01):
        """A completely connected linear layer.
        nn is a neural network that the inputs come from
        num_outputs is the number of outputs
        max_init is the maximum value for random initialization of parameters
        """
        # layer.ms[i][o] is running average of squared gradient input i and output o
        layer.ms = [[0 for _ in range(len(layer.weights[0]))]
                        for _ in range(len(layer.weights))]
        self.rho = rho
        self.epsilon = epsilon
        self.lr = lr
        
    def update(self, layer):
        """updates parameters after a batch"""
        for inp in range(len(layer.weights)):
            for out in range(len(layer.weights[0])):
                layer.ms[inp][out] = self.rho*layer.ms[inp][out]+ (1-self.rho) * layer.delta[inp][out]**2
                layer.weights[inp][out] -= self.lr * layer.delta[inp][out] / (layer.ms[inp][out]+self.epsilon)**0.5 
                layer.delta[inp][out] = 0
               
from utilities import flip
class Dropout_layer(Layer):
    """Dropout layer
    """
    
    def __init__(self, nn, rate=0):
        """
        rate is fraction of the input units to drop. 0 =< rate < 1
        """
        self.rate = rate
        Layer.__init__(self, nn)
        self.mask = [0]*self.num_inputs
        
    def output_values(self, input_values, training=False):
        """Returns the outputs for the input values.
        It remembers the input values and mask for the backprop.
        """
        if training:
            scaling = 1/(1-self.rate)
            for i in range(self.num_inputs):
                self.mask[i] = 0 if flip(self.rate) else 1
                input_values[i] = self.mask[i]*input_values[i]*scaling
        return input_values

    def backprop(self, output_errors):
        """Returns the derivative of the errors"""
        for i in range(self.num_inputs):
            self.input_errors[i] = output_errors[i]*self.mask[i]
        return self.input_errors


def main():
    """Sets up some global variables to allow for interaction
    """
    global data, nn3, nn3do
    #data = Data_from_file('data/mail_reading.csv', target_index=-1)
    #data = Data_from_file('data/mail_reading_consis.csv', target_index=-1)
    data = Data_from_file('data/SPECT.csv', target_index=0) #, seed=12345)
    #data = Data_from_file('data/carbool.csv', one_hot=True, target_index=-1, seed=123)
    #data = Data_from_file('data/iris.data', target_index=-1) 
    #data = Data_from_file('data/if_x_then_y_else_z.csv', num_train=8, target_index=-1) # not linearly sep
    #data = Data_from_file('data/holiday.csv', target_index=-1) #, num_train=19)
    #data = Data_from_file('data/processed.cleveland.data', target_index=-1)
    #random.seed(None)

    # nn3 is has a single hidden layer of width 3 
    nn3 = NN(data, optimizer=SGD)
    nn3.add_layer(Linear_complete_layer(nn3,3))
    #nn3.add_layer(Sigmoid_layer(nn3))  
    nn3.add_layer(ReLU_layer(nn3))
    nn3.add_layer(Linear_complete_layer(nn3, 1, final_layer=True)) # when output_type="boolean"
    print("nn3")
    nn3.learn(batch_size=100, num_iter = 1000, report_each=100)

    # Print some training examples
    #for eg in random.sample(data.train,10): print(eg,nn3.predictor(eg))

    # Print some test examples
    #for eg in random.sample(data.test,10): print(eg,nn3.predictor(eg))

    # To see the weights learned in linear layers
    # nn3.layers[0].weights
    # nn3.layers[2].weights

    # nn3do is like nn3 but with dropout on the hidden layer
    nn3do = NN(data, optimizer=SGD)
    nn3do.add_layer(Linear_complete_layer(nn3do,3))
    #nn3.add_layer(Sigmoid_layer(nn3))  # comment this or the next
    nn3do.add_layer(ReLU_layer(nn3do))
    nn3do.add_layer(Dropout_layer(nn3do, rate=0.5))
    nn3do.add_layer(Linear_complete_layer(nn3do, 1, final_layer=True))
    #nn3do.learn(batch_size=100, num_iter = 1000, report_each=100)

if __name__ == "__main__":
    main()
    
class NN_from_arch(NN):
    def __init__(self, data, arch, optimizer=SGD, **hyperparms):
        """arch is a list of widths of the hidden layers from bottom up.
        opt is an optimizer (one of: SGD, Momentum, RMS_Prop)
        hyperparms is the parameters of the optimizer 
        returns a neural network with ReLU activations on hidden layers
        """
        NN.__init__(self, data, optimizer=optimizer, **hyperparms)
        for width in arch:
            self.add_layer(Linear_complete_layer(self,width))
            self.add_layer(ReLU_layer(self))
        output_size = len(data.target.frange) if data.target.ftype == "categorical" else 1
        self.add_layer(Linear_complete_layer(self,output_size, final_layer=True))
        hyperparms_string = ','.join(f"{p}={v}" for p,v in hyperparms.items())
        self.name = f"NN({arch},{optimizer.__name__}({hyperparms_string}))"

    def __str__(self):
         return self.name

# nn3a = NN_from_arch(data, [3], SGD, lr=0.001)

from learnLinear import plot_steps
from learnProblem import Evaluate

# To show plots first choose a criterion to use
crit = Evaluate.log_loss # penalizes overconfident predictions (when wrong)
# crit = Evaluate.accuracy  # only considers mode
# crit = Evaluate.squared_loss  # penalizes overconfident predictions less

def plot_algs(data, archs=[[3]], opts=[SGD],lrs=[0.1, 0.01,0.001,0.0001],
                    criterion=crit,  num_steps=1000):
    args = []
    for arch in archs:
        for opt in opts:
            for lr in lrs:
                args.append((arch,opt,{'lr':lr}))
    plot_algs_opts(data, args, criterion,  num_steps)

def plot_algs_opts(data, args, criterion=crit,  num_steps=1000):
    """args is a list of (architecture, optimizer, parameters)
       for each of the corresponding triples it plots the learning rate"""
    for (arch, opt, hyperparms) in args:
        nn = NN_from_arch(data, arch, opt, **hyperparms)
        plot_steps(data, learner = nn, criterion=crit, num_steps=num_steps,
                       log_scale=False, legend_label=str(nn))

## first select good learning rates for each optimizer.
# plot_algs(data, archs=[[3]], opts=[SGD],lrs=[0.1, 0.01,0.001,0.0001])
# plot_algs(data, archs=[[3]], opts=[Momentum],lrs=[0.1, 0.01,0.001,0.0001])
# plot_algs(data, archs=[[3]], opts=[RMS_Prop],lrs=[0.1, 0.01,0.001,0.0001])

## If they have the same best learning rate, compare the optimizers:
# plot_algs(data, archs=[[3]], opts=[SGD,Momentum,RMS_Prop],lrs=[0.01])

## With different learning rates, compare the optimizer using:
# plot_algs_opts(data, args=[([3],SGD,{'lr':0.01}), ([3],Momentum,{'lr':0.1}), ([3],RMS_Prop,{'lr':0.001})])

# similarly select the best architecture, but the best learning rate might depend also on the architecture

# Simplified version: (approx 6000 training instances)
# data_mnist = Data_from_file('../MNIST/mnist_train.csv', prob_test=0.9, target_index=0, target_type="categorical")

# Full version:
# data_mnist = Data_from_files('../MNIST/mnist_train.csv', '../MNIST/mnist_test.csv', target_index=0,  target_type="categorical")

#nn_mnist = NN_from_arch(data_mnist, [32,10], SGD, lr=0.01})
# one epoch:
# start_time = time.perf_counter();nn_mnist.learn(batch_size=128, num_iter=len(data_mnist)/128 );end_time = time.perf_counter();print("Time:", end_time - start_time,"seconds") 
# determine train error:
# data_mnist.evaluate_dataset(data_mnist.train, nn_mnist.predictor, Evaluate.accuracy)
# determine test error:
# data_mnist.evaluate_dataset(data_mnist.test, nn_mnist.predictor, Evaluate.accuracy)
# Print some random predictions:
# for eg in random.sample(data_mnist.test,10): print(data_mnist.target(eg), nn_mnist.predictor(eg), nn_mnist.predictor(eg)[data_mnist.target(eg)])
# Plot learning:
# plot_algs(data_mnist,archs=[[32],[32,8]], opts=[RMS_Prop], lrs=[0.01], data=data_mnist, num_steps=100)
# plot_algs(data_mnist,archs=[[8],[8,8,8],[8,8,8,8,8,8,8]], opts=[RMS_Prop], lrs=[0.01], data=data_mnist, num_steps=100)

