#cython: language_level=3

# Be careful, order is important ! if you need a module, it has to be imported before

# fused types for dataset readers
ctypedef fused Reader:
    PyMnistReader
    PyTextReader

# import datasets
include "datasets/mnist_reader.pyx"
include "datasets/text_reader.pyx"

# import networks, based on datasets
include "networks/dense_dense_net.pyx"
include "networks/dense_dense_dense_net.pyx"
include "networks/lenet.pyx"
include "networks/alexnet.pyx"
include "networks/vggnet16.pyx"
