# distutils: language = c++
# distutils: sources = ../dll/release/lib/lib_dll_mnist_mylib.so
from mnist_lib cimport *

# Class for getting dataset and training/evaluating a neural network
cdef class PyMnistLib():
    # Create instance of MnistLib
    cdef MnistLib *ptr

    # Constructor
    def __cinit__(self):
       self.ptr = new MnistLib()

    # Get dataset
    def get_dataset(self):
        return self.ptr.getDataset()


# Execute simple example
def do_simple_example():
    doSimpleExample()