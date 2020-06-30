from libcpp.vector cimport vector
from libc.stdint cimport uint8_t
from libcpp.string cimport string


cdef extern from "../../dll/python-wrapper-lib/datasets/MnistReader.h":
    cdef struct MnistDataset:
        vector[vector[uint8_t]] training_images
        vector[vector[uint8_t]] test_images
        vector[uint8_t] training_labels
        vector[uint8_t] test_labels

    # Class for getting dataset
    cdef cppclass MnistReader:
        MnistReader()

        MnistDataset& readDataset()

        void display()

        void displayPretty()
