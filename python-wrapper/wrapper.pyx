# distutils: language = c++
# distutils: sources = ../dll/release/lib/libdll_mnist_mylib.so
from mnist_lib cimport *

# Class for getting dataset and training/evaluating a neural network
cdef class PyMnistLib():
    # Create instance of MnistLib
    cdef MnistLib *ptr

    # Constructor
    def __cinit__(self):
       self.ptr = new MnistLib()

    # Get dataset
    def get_dataset(self):
        return self.ptr.getDataset()

    def create_net(self, size_t nb_visibles, size_t nb_hiddens, size_t learning_rate):
        return self.ptr.createNet(nb_visibles, nb_hiddens,learning_rate)

    def display_dataset(self):
        return self.ptr.displayDataset()

    def display_dataset_pretty(self):
        return self.ptr.displayDatasetPretty()

    def display_net(self):
        return self.ptr.displayNet()

    def train(self, size_t epochs):
        return self.ptr.train(epochs)

    def evaluate(self):
        return self.ptr.evaluate()




# Execute simple example
def do_simple_example():
    doSimpleExample()