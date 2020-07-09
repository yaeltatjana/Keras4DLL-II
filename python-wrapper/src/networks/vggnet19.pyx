from src.networks.vggnet19 cimport *
from cython.operator cimport dereference as deref

cdef class PyVGGNet19:
    # Create instance of VGGNet
    cdef VGGNet19 *ptr

    # Constructor
    def __cinit__(self):
        self.ptr = new VGGNet19()

    def __dealloc__(self):
        del self.ptr

    def display(self):
        self.ptr.display()

    def display_pretty(self):
        self.ptr.displayPretty()

    def set_conv_layer(self, size_t layer, size_t channels, size_t dim1, size_t dim2, size_t nb_filters, size_t filt1, size_t filt2):
        self.ptr.setConvLayer(layer, channels, dim1, dim2, nb_filters, filt1, filt2)

    def set_mp_layer(self, size_t layer, size_t channels, size_t dim1, size_t dim2, size_t pool1, size_t pool2):
        self.ptr.setMPLayer(layer, channels, dim1, dim2, pool1, pool2)

    def set_dense_layer(self, size_t layer, size_t input_size, size_t output_size):
        self.ptr.setDenseLayer(layer, input_size, output_size)

    def set_learning_rate(self, double rate):
        self.ptr.setLearningRate(rate)

    def set_adam_beta1(self, double beta):
        self.ptr.setAdamBeta1(beta)

    def set_adam_beta2(self, double beta):
        self.ptr.setAdamBeta2(beta)

    def fine_tune(self, Reader ds, size_t epochs):
        return self.ptr.fineTune(deref(ds.ptr), epochs)

    def evaluate(self, Reader ds):
        self.ptr.evaluate(deref(ds.ptr))

    def store_weights(self, str file):
        self.ptr.storeWeights(file.encode('utf-8'))

    def load_weights(self, str file):
        self.ptr.loadWeights(file.encode('utf-8'))
