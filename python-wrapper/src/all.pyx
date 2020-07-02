#cython: language_level=3

include "./datasets/mnist_reader.pyx"
include "./datasets/mnist_3d_reader.pyx"
include "./datasets/text_reader.pyx"

include "./networks/dense_dense_net.pyx"
include "./networks/dense_dense_dense_net.pyx"
include "./networks/lenet.pyx"
include "./networks/alexnet.pyx"
include "./networks/vggnet.pyx"

