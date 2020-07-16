from src.datasets.text_reader cimport *

cdef class PyTextReader:
    """
    Class to read a dataset from a text file
    Attributes:
        ptr     Instance of corresponding C++ object
    """
    cdef TextReader *ptr

    def __cinit__(self, str img_path, str lbl_path, size_t img_limit = 0, size_t lbl_lLimit = 0):
        """
        Constructor method that allocates memory for the C++ object

        :param img_path:    Path to images' text file
        :type img_path:     str
        :param lbl_path:    Path to labels' text file
        :type lbl_path:     str
        :param img_limit:   Size limit of image samples to read
        :type img_limit:    str
        :param lbl_lLimit:  Size limit of label samples to read
        :type lbl_lLimit:   str
        :return:            instance of class
        :rtype:             PyTextReader
        """
        self.ptr = new TextReader(img_path.encode('utf-8'), lbl_path.encode('utf-8'), img_limit, lbl_lLimit)

    def __dealloc__(self):
        """
        Destructor function that deallocates memory of C++ object.
        """
        del self.ptr

    def read_images(self):
        """
        Read images of dataset
        :return:    Images
        :rtype:     list
        """
        return self.ptr.readImages()

    def read_labels(self):
        """
        Read labels of dataset
        :return:    Labels
        :rtype:     list
        """
        return self.ptr.readLabels()
