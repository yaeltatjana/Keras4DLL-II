from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t

cdef extern from "../dll/python-wrapper-lib/mnist_lib.h":
    # MNIST dataset structure
    cdef struct Dataset:
        vector[vector[uint8_t]] training_images
        vector[vector[uint8_t]] test_images
        vector[uint8_t] training_labels
        vector[uint8_t] test_labels

    # Class for getting dataset and training/evaluating a neural network
    cdef cppclass MnistLib:
        MnistLib()

        Dataset getDataset()

        void createNet(size_t nb_visibles, size_t nb_hiddens, size_t learning_rate)

        void displayNet()

        void displayDataset()

        void displayDatasetPretty()

        float train(size_t epochs)

        void evaluate()


    # Function that exectues the simple example
    cdef void doSimpleExample() nogil
