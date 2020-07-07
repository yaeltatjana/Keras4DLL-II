from src.networks.lenet cimport *
from cython.operator cimport dereference as deref

cdef class PyLeNet:
    # Create instance of LeNet
    cdef LeNet *ptr

    # Constructor
    def __cinit__(self):
        self.ptr = new LeNet()

    def __dealloc__(self):
        del self.ptr

    def display(self):
        self.ptr.display()

    def display_pretty(self):
        self.ptr.displayPretty()

    def set_conv_layer(self, size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t nbFilters, size_t firstDimFilter, size_t secDimFilter):
        self.ptr.setConvLayer(layer, inputChannels, firstDim, secDim, nbFilters, firstDimFilter, secDimFilter)

    def set_mp_layer(self, size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t firstDimPoolSize, size_t secDimPoolSize):
        self.ptr.setMPLayer(layer, inputChannels, firstDim, secDim, firstDimPoolSize, secDimPoolSize)

    def set_dense_layer(self, size_t layer, size_t inputSize, size_t outputSize):
        self.ptr.setDenseLayer(layer, inputSize, outputSize)

    def set_learning_rate(self, double rate):
        self.ptr.setLearningRate(rate)

    def set_adam_beta1(self, double beta):
        self.ptr.setAdamBeta1(beta)

    def set_adam_beta2(self, double beta):
        self.ptr.setAdamBeta2(beta)

    def fine_tune(self, PyMnistReader ds, size_t epochs):
        self.ptr.fineTune(deref(ds.ptr), epochs)

    def evaluate(self, PyMnistReader ds):
        self.ptr.evaluate(deref(ds.ptr))

    def store_weights(self, str file):
        self.ptr.storeWeights(file.encode('utf-8'))

    def load_weights(self, str file):
        self.ptr.loadWeights(file.encode('utf-8'))
