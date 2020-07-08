#cython: language_level=3

# Be careful, order is important ! if you need a module, it has to be imported before

# first import fused types
include "common.pyx"

# import datasets
include "datasets/mnist_reader.pyx"
include "datasets/mnist_3d_reader.pyx"
include "datasets/text_reader.pyx"

# import networks, based on datasets
include "networks/dense_dense_net.pyx"
include "networks/dense_dense_dense_net.pyx"
include "networks/lenet.pyx"
include "networks/alexnet.pyx"
include "networks/vggnet19.pyx"

