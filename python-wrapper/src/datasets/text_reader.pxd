from libcpp.vector cimport vector
from libc.stdint cimport uint8_t
from libcpp.string cimport string


cdef extern from "../../dll/python-wrapper-lib/datasets/TextReader.h":
    cdef cppclass TextReader:
        TextReader(string imgsPath, string labelsPath, size_t imgLimit, size_t labelLimit)

        vector[vector[uint8_t]] readImages()

        vector[uint8_t] readLabels()
