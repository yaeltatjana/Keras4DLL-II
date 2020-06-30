from libcpp.vector cimport vector
from libcpp.string cimport string
from src.datasets.mnist_reader cimport *

# path is the one from src/all.pyx as it import this file after
cdef extern from "../../dll/python-wrapper-lib/networks/DenseDenseDenseNet.h":
    cdef cppclass DenseDenseDenseNet:
        DenseDenseDenseNet()

        DenseDenseDenseNet(vector[size_t]& nb_input, vector[size_t]& nb_output)

        void setLearningRate(double l_rate)

        void setLayerSize(size_t layer, size_t input_size, size_t output_size)

        void setInitialMomentum(double m)

        void display()

        void displayPretty()

        float fineTune(MnistReader& ds, size_t epochs)

        void evaluate(MnistReader& ds)

        void storeWeights(const string& file)

        void loadWeights(const string& file)
