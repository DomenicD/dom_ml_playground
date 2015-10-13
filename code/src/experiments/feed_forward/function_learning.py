import math;
from code.src.learning_algorithms.backpropagation import Backpropagator
from code.src.data_processing.expectation import Expectation
from code.src.data_processing.normalizer import Normalizer
from code.src.neural_networks.feed_forward import FeedForwardNN



    
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

    results = backpropagator.teach(
        network, expectations, learning_rate = 1.5, max_iterations = 10000,
        acceptable_error = 1, callback_rate = 100,
        callback_func = monitor_neural_network)

    return network
        

# TODO(domenicd): Need to plot these to see what is going on. The errors are
#                 super large right now e+11 was the lowest achived in 10000
#                 iterations.
def monitor_neural_network(neural_network, expectations, training_result):
    print(training_result.epochs)
    print(training_result.error)


# Have to add location of git project to PYTHONPATH system variable if you
# want to run this script directly.
if __name__ == '__main__':
    fn = (lambda x: [math.pow(x, 4) + math.pow(x, 3) + math.pow(x, 2) + 
          x + 4])
    learn_function(fn, -50, 50)