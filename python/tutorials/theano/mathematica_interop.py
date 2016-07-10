import numpy
from logistic_regression import LogisticRegression, load_data
import multilayer_perceptron as mp

def get_data(dataset):
    datasets = load_data(dataset)    
    test_set_x, test_set_y = datasets[2]    
    numpy.savetxt('test_set_x.txt', test_set_x.get_value())
    numpy.savetxt('test_set_y.txt', test_set_y.eval())


if __name__ == '__main__':
    get_data('D:\git\DeepLearningTutorials\data\mnist.pkl.gz')