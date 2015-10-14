﻿import math
import matplotlib.pyplot as plt
import threading
from code.src.learning_algorithms.backpropagation import Backpropagator
from code.src.data_processing.expectation import Expectation
from code.src.data_processing.normalizer import Normalizer
from code.src.neural_networks.feed_forward import FeedForwardNN
from code.src.neural_networks.neural_network_utils import NeuralNetworkUtils



    
def learn_function(fn, in_min, in_max):
    expectations = [Expectation([i], fn(i))
                    for i in range(in_min, in_max + 1)]

    in_range = in_max - in_min
    in_min = in_min - .1 * in_range
    in_max = in_max + .1 * in_range
        
    out_min = None
    out_max = None

    for e in expectations:
        if out_min == None or e.outputs[0] < out_min:
            out_min = e.outputs[0]

        if out_max == None or e.outputs[0] > out_max:
            out_max = e.outputs[0]


    out_range = out_max - out_min
    out_min = out_min - .1 * out_range
    out_max = out_max + .1 * out_range

    backpropagator = Backpropagator()
        
    normalizer = Normalizer(in_min = in_min, in_max = in_max,
                            out_min = out_min, out_max = out_max,
                            norm_min = -2, norm_max = 2)
        
    network = FeedForwardNN(normalizer, [1, 3, 1])
    network.randomize_connection_weights()

    
    config_fit_tracker(network, expectations)

    # TODO(domenicd): There are some issues between updating the plot and
    #                 the data in different threads. The plot is not movable
    #                 for some reason.
    training_thread = threading.Thread(target=lambda: backpropagator.teach(
        network, expectations, learning_rate = 1.5, max_iterations = 10000,
        acceptable_error = 1, callback_func = update_fit_plot))
    training_thread.daemon = True
    training_thread.start()

    plt.figure('Fit Tracker').show()

    training_thread.join()
    return network
        

def config_fit_tracker(neural_network, expectations):
    x = [e.inputs[0] for e in expectations]
    goal = [e.outputs[0] for e in expectations]
    current = NeuralNetworkUtils.list_of_outputs(neural_network,
        [e.inputs for e in expectations])
    
    fig = plt.figure('Fit Tracker')    
    ax = fig.add_subplot(111)
    ax.plot(x, goal)
    ax.plot(x, current)
    fig.canvas.draw()

    
def update_fit_plot(neural_network, expectations, training_result):
    fig = plt.figure('Fit Tracker')
    if len(fig.axes) > 0 and len(fig.axes[0].lines) > 1:
        ax = fig.axes[0]
        line = ax.lines[1]
        line.set_ydata(NeuralNetworkUtils.list_of_outputs(neural_network,
            [e.inputs for e in expectations]))
        fig.canvas.draw()


# Please see README.md if you are having trouble getting this to run.
if __name__ == '__main__':
    fn = (lambda x: [math.pow(x, 4) + math.pow(x, 3) + math.pow(x, 2) + 
          x + 4])
    learn_function(fn, -50, 50)