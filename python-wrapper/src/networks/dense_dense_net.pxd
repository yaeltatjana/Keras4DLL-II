from libcpp.vector cimport vector
from libcpp.string cimport string
from src.datasets.mnist_reader cimport *

cdef extern from "../../dll/python-wrapper-lib/DenseDenseNet.h":
    # Class for getting dataset
    cdef cppclass DenseDenseNet:
        DenseDenseNet()

        DenseDenseNet(vector[size_t]& nb_input, vector[size_t]& nb_output)

        void setLearningRate(double l_rate)

        void setLayerSize(size_t layer, size_t input_size, size_t output_size)

        void setInitialMomentum(double m)

        void display()

        void displayPretty()

        float fineTune(MnistReader& ds, size_t epochs)

        void evaluate(MnistReader& ds)

        void storeWeights(string &file)

        void loadWeights(string &file)