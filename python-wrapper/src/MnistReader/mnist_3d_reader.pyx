from src.MnistReader.mnist_3d_reader cimport *

cdef class PyMnist3DReader:
    # Create instance of MnistReader
    cdef Mnist3DReader *ptr

    # Constructor
    def __cinit__(self):
        self.ptr = new Mnist3DReader()

    def read_training_images(self):
        return self.ptr.readTrainingImages()

    def read_test_images(self):
        return self.ptr.readTestImages()

    def read_training_labels(self):
        return self.ptr.readTrainingLabels()

    def read_test_labels(self):
        return self.ptr.readTestLabels()