from libcpp.vector cimport vector
from src.networks.dense_dense_net cimport *
from cython.operator cimport dereference as deref

cdef class PyDenseDenseNet:
    # Create instance of DenseDenseNet
    cdef DenseDenseNet *ptr

    # Constructor
    def __cinit__(self, vector[size_t]& nb_input = vector[size_t](), vector[size_t]& nb_output = vector[size_t]()):
        if (nb_input.size() == 0 or nb_output.size() == 0):
            self.ptr = new DenseDenseNet()
        else:
            self.ptr = new DenseDenseNet(nb_input, nb_output)

    def __dealloc__(self):
        del self.ptr

    def set_learning_rate(self, double l_rate):
        self.ptr.setLearningRate(l_rate)

    def set_layer_size(self, size_t layer, size_t input_size, size_t output_size):
        self.ptr.setLayerSize(layer, input_size, output_size)

    def set_initial_momentum(self, double m):
        self.ptr.setInitialMomentum(m)

    def display(self):
        self.ptr.display()

    def display_pretty(self):
        self.ptr.displayPretty()

    def fine_tune(self, PyMnistReader ds, size_t epochs):
        self.ptr.fineTune(deref(ds.ptr), epochs)

    def evaluate(self, PyMnistReader ds):
        self.ptr.evaluate(deref(ds.ptr))

    def store_weights(self, str file):
        self.ptr.storeWeights(file.encode('utf-8'))

    def load_weights(self, str file):
        self.ptr.loadWeights(file.encode('utf-8'))