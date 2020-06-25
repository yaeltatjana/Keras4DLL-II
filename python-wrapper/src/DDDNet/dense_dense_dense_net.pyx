from libcpp.vector cimport vector
from src.DDDNet.dense_dense_dense_net cimport *
from cython.operator cimport dereference as deref

cdef class PyDenseDenseDenseNet:
    # Create instance of DenseDenseDenseNet
    cdef DenseDenseDenseNet *ptr

    # Constructor
    #def __cinit__(self):
        #self.ptr = new DenseDenseDenseNet()

    # Constructor
    def __cinit__(self, vector[size_t]& nb_input = vector[size_t](), vector[size_t]& nb_output = vector[size_t]()):
        if (nb_input.size() == 0 or nb_output.size() == 0):
            self.ptr = new DenseDenseDenseNet()
        else:
            self.ptr = new DenseDenseDenseNet(nb_input, nb_output)

    def setLearningRate(self, double l_rate):
        self.ptr.setLearningRate(l_rate)

    def setLayerSize(self, size_t layer, size_t input_size, size_t output_size):
        self.ptr.setLayerSize(layer, input_size, output_size)

    def setInitialMomentum(self, double m):
        self.ptr.setInitialMomentum(m)

    def display(self):
        self.ptr.display()

    def display_pretty(self):
        self.ptr.displayPretty()

    def fineTune(self, PyMnistReader ds, size_t epochs):
        self.ptr.fineTune(deref(ds.ptr), epochs)

    def evaluate(self, PyMnistReader ds):
        self.ptr.evaluate(deref(ds.ptr))

    def store_weights(self, str file):
        self.ptr.storeWeights(file.encode('utf-8'))

    def load_weights(self, str file):
        self.ptr.loadWeights(file.encode('utf-8'))


