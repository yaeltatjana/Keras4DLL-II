from libcpp.string cimport string
from src.datasets.mnist_reader cimport *
from src.datasets.text_reader cimport *

cdef extern from "../../dll/python-wrapper-lib/networks/AlexNet.h":
    cdef cppclass AlexNet:
        AlexNet()

        void display()

        void displayPretty()

        void setConvLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t nbFilters, size_t firstDimFilter, size_t secDimFilter)

        void setMPLayer(size_t layer, size_t inputChannels, size_t firstDim, size_t secDim, size_t firstDimPoolSize, size_t secDimPoolSize)

        void setDenseLayer(size_t layer, size_t inputSize, size_t outputSize)

        void setLearningRate(double rate)

        void setAdamBeta1(double beta)

        void setAdamBeta2(double beta)

        float fineTune(MnistReader& ds, size_t epochs)

        void evaluate(MnistReader& ds)

        float fineTune(TextReader& ds, size_t epochs)

        void evaluate(TextReader& ds)

        void storeWeights(string &file)

        void loadWeights(string &file)

