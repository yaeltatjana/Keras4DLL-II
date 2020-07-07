from src.datasets.text_reader cimport *

cdef class PyTextReader:
    cdef TextReader *ptr

    # Constructor
    def __cinit__(self, str img_path, str lbl_path, size_t img_limit = 0, size_t lbl_lLimit = 0):
        self.ptr = new TextReader(img_path.encode('utf-8'), lbl_path.encode('utf-8'), img_limit, lbl_lLimit)

    def __dealloc__(self):
        del self.ptr

    def read_images(self):
        return self.ptr.readImages()

    def read_labels(self):
        return self.ptr.readLabels()