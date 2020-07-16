from src.datasets.mnist_reader cimport *

cdef class PyMnistReader:
    """
    Class that represents the reader for the MNIST dataset.

    Attributes:
        ptr     instance of C++ MNISTReader
    """
    cdef MnistReader *ptr

    def __cinit__(self):
        """
        Construct a PyMnistReader instance by allocating memory
        :return:    instance of class
        :rtype:     PyMnistReader
        """
        self.ptr = new MnistReader()

    def __dealloc__(self):
        """
        Destructor to deallocate memory of C++ object
        """
        del self.ptr

    def read_dataset(self):
        """
        Read the entire dataset
        :return:    MNIST Dataset with distinction between images and labels for train and test set
        :rtype:     dict
        """
        return self.ptr.readDataset()

    def display(self):
        """
        Display MNIST dataset
        """
        self.ptr.display()

    def display_pretty(self):
        """
        Display MNIST Dataset in the pretty form
        """
        self.ptr.displayPretty()
