from src.networks.vggnet16 cimport *
from cython.operator cimport dereference as deref

cdef class PyVGGNet16:
    """
    Class to create an VGGNet16 network
    Attributes:
        ptr     corresponding C++ instance
    """
    cdef VGGNet16 *ptr

    def __cinit__(self):
        """
        Constructor to allocate memory for the attribute
        :return:    instance of this class
        :rtype:     PyVGGNet16
        """
        self.ptr = new VGGNet16()

    def __dealloc__(self):
        """
        Destructor to deallocate memory of attribute
        """
        del self.ptr

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

    def set_conv_layer(self, size_t layer, size_t channels, size_t dim1, size_t dim2, size_t nb_filters, size_t filt1,
                       size_t filt2):
        """
        Initialize a convolutional layer
        :param layer:               index of layer
        :type layer:                int
        :param inputChannels:       number of inputs channels
        :type inputChannels:        int
        :param firstDim:            first dimension of input image
        :type firstDim:             int
        :param secDim:              second dimension of input image
        :type secDim:               int
        :param nbFilters:           number of filters
        :type nbFilters:            int
        :param firstDimFilter:      first dimension of filter
        :type firstDimFilter:       int
        :param secDimFilter:        second dimension of filter
        :type secDimFilter:         int
        """
        self.ptr.setConvLayer(layer, channels, dim1, dim2, nb_filters, filt1, filt2)

    def set_mp_layer(self, size_t layer, size_t channels, size_t dim1, size_t dim2, size_t pool1, size_t pool2):
        """
        Initialize max_pooling layer
        :param layer:               index of layer
        :type layer:                int
        :param inputChannels:       number of input channels
        :type inputChannels:        int
        :param firstDim:            first dimension of input image
        :type firstDim:             int
        :param secDim:              second dimension of input image
        :type secDim:               int
        :param firstDimPoolSize:    first dimension of pooling kernel
        :type firstDimPoolSize:     int
        :param secDimPoolSize:      second dimension of pooling kernel
        :type secDimPoolSize:       int
        """
        self.ptr.setMPLayer(layer, channels, dim1, dim2, pool1, pool2)

    def set_dense_layer(self, size_t layer, size_t input_size, size_t output_size):
        """
        Initialize a dense layer
        :param layer:       index of layer
        :type layer:        int
        :param inputSize:   input size of layer
        :type inputSize:    int
        :param outputSize:  output size of layer
        :type outputSize:   int
        """
        self.ptr.setDenseLayer(layer, input_size, output_size)

    def set_learning_rate(self, double rate):
        """
        Change learning rate for training
        :param rate:    learning rate
        :type rate:     double
        """
        self.ptr.setLearningRate(rate)

    def set_adam_beta1(self, double beta):
        """
        Define adam_bet1 parameter
        :param beta:    value of parameter
        :type beta:     double
        """
        self.ptr.setAdamBeta1(beta)

    def set_adam_beta2(self, double beta):
        """
        Define adam_bet2 parameter
        :param beta:    value of parameter
        :type beta:     double
        """
        self.ptr.setAdamBeta2(beta)

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
