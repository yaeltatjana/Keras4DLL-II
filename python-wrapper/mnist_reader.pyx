# distutils: language = c++
# distutils: sources = ../dll/release/lib/libdll_mnist_mylib.so
from mnist_reader cimport *

#
cdef class PyMnistReader():
    # Create instance of MnistReader
    cdef MnistReader *ptr

    # Constructor
    def __cinit__(self):
        self.ptr = new MnistReader()

    def display(self):
        self.ptr.display()

    def display_pretty(self):
        self.ptr.displayPretty()