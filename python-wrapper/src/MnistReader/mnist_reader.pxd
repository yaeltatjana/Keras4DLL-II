from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t

cdef extern from "../../dll/python-wrapper-lib/MnistReader.h":

    # Class for getting dataset
    cdef cppclass MnistReader:
        MnistReader()

        void display()

        void displayPretty()