from src.datasets.mnist_reader cimport *

cdef class PyMnistReader:
    # Create instance of MnistReader
    cdef MnistReader *ptr

    # Constructor
    def __cinit__(self):
        self.ptr = new MnistReader()

    def __dealloc__(self):
        del self.ptr

    def read_dataset(self):
        return self.ptr.readDataset()

    def display(self):
        self.ptr.display()

    def display_pretty(self):
        self.ptr.displayPretty()
