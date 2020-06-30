from libcpp.vector cimport vector
from libc.stdint cimport uint8_t

cdef extern from "../../dll/python-wrapper-lib/Mnist3DReader.h":
    # Class for getting dataset
    cdef cppclass Mnist3DReader:
        Mnist3DReader()
        vector [vector [uint8_t]] readTrainingImages()
        vector [vector [uint8_t]] readTestImages()
        vector [uint8_t] readTrainingLabels()
        vector [uint8_t] readTestLabels()