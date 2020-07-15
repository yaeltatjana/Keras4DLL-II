from libcpp.vector cimport vector
from src.networks.dense_dense_net cimport *
from cython.operator cimport dereference as deref

cdef class PyDenseDenseNet:
    """
    Class to create an DenseDenseNet network
    Attributes:
        ptr     corresponding C++ instance
    """
    cdef DenseDenseNet *ptr

    def __cinit__(self, vector[size_t]& nb_input = vector[size_t](), vector[size_t]& nb_output = vector[size_t]()):
        """
        Constructor to allocate memory for the attribute.

        :param nb_input:    vector containing input size of layers
        :type nb_input:     list
        :param nb_output:   vector containing output size of layers
        :type nb_output:    list
        """
        if (nb_input.size() == 0 or nb_output.size() == 0):
            self.ptr = new DenseDenseNet()
        else:
            self.ptr = new DenseDenseNet(nb_input, nb_output)

    def __dealloc__(self):
        """
        Destructor to deallocate memory of attribute
        """
        del self.ptr

    def set_learning_rate(self, double rate):
        """
        Change learning rate for training
        :param rate:    learning rate
        :type rate:     double
        """
        self.ptr.setLearningRate(rate)

    def set_layer_size(self, size_t layer, size_t input_size, size_t output_size):
        """
        Set size of dense layers
        :param layer:       index of layer
        :type layer:        int
        :param input_size:  input size
        :type input_size:   int
        :param output_size: output size
        :type output_size:  int
        """
        self.ptr.setLayerSize(layer, input_size, output_size)

    def set_initial_momentum(self, double m):
        """
        Change initial momentum parameter
        :param m:   initial momentum
        :type m:    double
        """
        self.ptr.setInitialMomentum(m)

    def display(self):
        """
        Display network
        """
        self.ptr.display()

    def display_pretty(self):
        """
        Display network in the pretty form
        """
        self.ptr.displayPretty()

    def fine_tune(self, Reader ds, size_t epochs):
        """
        Train network with dataset contained in the reader
        :param ds:          reader containing dataset
        :type ds:           Reader
        :param epochs:      number of epochs
        :type epochs:       int
        :return:            classification error
        :rtype:             float
        """
        return self.ptr.fineTune(deref(ds.ptr), epochs)

    def evaluate(self, Reader ds):
        """
        Inference of a the network
        :param ds:      reader containing dataset
        :type ds:       Reader
        """
        self.ptr.evaluate(deref(ds.ptr))

    def store_weights(self, str file):
        """
        Store weights from network into a file
        :param file:    filename
        :type file:     str
        """
        self.ptr.storeWeights(file.encode('utf-8'))

    def load_weights(self, str file):
        """
        Load weights into network from a file
        :param file:    filename
        :type file:     str
        """
        self.ptr.loadWeights(file.encode('utf-8'))