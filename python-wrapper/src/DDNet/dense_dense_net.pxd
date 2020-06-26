from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
#from libc.stdint cimport uint8_t
from src.MnistReader.mnist_reader cimport *

#cdef extern from "../../dll/python-wrapper-lib/MnistReader.h":

    # Class for getting dataset
    #cdef cppclass MnistReader:
        #MnistReader()

        #void display()

        #void displayPretty()

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

        void all()