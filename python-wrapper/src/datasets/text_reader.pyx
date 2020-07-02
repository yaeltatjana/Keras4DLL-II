from src.datasets.text_reader cimport *

cdef class PyTextReader:
    cdef TextReader *ptr

    # Constructor
    def __cinit__(self, str imgPath, str lblPath, size_t imgLimit = 0, size_t lblLimit = 0):
        self.ptr = new TextReader(imgPath.encode('utf-8'), lblPath.encode('utf-8'), imgLimit, lblLimit)

    def read_images(self):
        return self.ptr.readImages()

    def read_labels(self):
        return self.ptr.readLabels()