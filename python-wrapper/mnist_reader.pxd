from libcpp.vector cimport vector
from libc.stdint cimport uint8_t


cdef struct mystruct:
    vector[vector[uint8_t]] training_images
    vector[vector[uint8_t]] test_images
    vector[uint8_t] training_labels
    vector[uint8_t] test_labels

#cdef extern from "../dll/python-wrapper-lib/MnistReader.h" namespace "mnist":
    #cdef cppclass MNIST_dataset[Container, Pixel, Label]:
        #pass
        #Container[Container[Pixel]] training_images
        #Container[Container[Pixel]] test_images
        #Container[Label] training_labels
        #Container[Label] test_labels

    #cdef cppclass MNIST_dataset[vector,uint8_t,uint8_t]:
        #vector[vector[uint8_t]] training_images
        #vector[vector[uint8_t]] test_images
        #vector[uint8_t] training_labels
        #vector[uint8_t] test_labels


cdef extern from "../dll/python-wrapper-lib/MnistReader.h":

    # Class for getting dataset
    cdef cppclass MnistReader:
        MnistReader()

        void display()

        void displayPretty()

        #MNIST_dataset[vector[uint8_t],uint8_t,uint8_t] readDataset()
        #MNIST_dataset readDataset()

