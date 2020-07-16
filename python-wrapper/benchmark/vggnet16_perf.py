import dll
import time
from benchmark.common import *


@benchmark("VGGNet16")
def perf_init(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyVGGNet16()
        net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
        net.set_conv_layer(1, 12, 24, 24, 12, 1, 1)
        net.set_mp_layer(2, 12, 24, 24, 2, 2)

        net.set_conv_layer(3, 12, 12, 12, 24, 3, 3)
        net.set_conv_layer(4, 24, 10, 10, 24, 1, 1)
        net.set_mp_layer(5, 24, 10, 10, 2, 2)

        net.set_conv_layer(6, 24, 5, 5, 24, 1, 1)
        net.set_conv_layer(7, 24, 5, 5, 24, 1, 1)
        net.set_conv_layer(8, 24, 5, 5, 24, 2, 2)
        net.set_mp_layer(9, 24, 4, 4, 2, 2)

        net.set_conv_layer(10, 24, 4, 4, 24, 1, 1)
        net.set_conv_layer(11, 24, 4, 4, 24, 1, 1)
        net.set_conv_layer(12, 24, 4, 4, 24, 1, 1)
        net.set_mp_layer(13, 24, 4, 4, 2, 2)

        net.set_conv_layer(14, 24, 2, 2, 24, 1, 1)
        net.set_conv_layer(15, 24, 2, 2, 24, 1, 1)
        net.set_conv_layer(16, 24, 2, 2, 24, 1, 1)
        net.set_mp_layer(17, 24, 2, 2, 2, 2)

        net.set_dense_layer(18, 24 * 1 * 1, 16)
        net.set_dense_layer(19, 16, 12)
        net.set_dense_layer(20, 12, 10)

        net.set_learning_rate(0.1)
        net.set_adam_beta1(0.997)
        net.set_adam_beta2(0.997)

        end = time.time()
        t.append(end - start)
    return t


@benchmark("VGGNet16")
def perf_display(loops, net):
    return perf_display_generic(loops, net)


@benchmark("VGGNet16")
def perf_display_pretty(loops, net):
    return perf_display_pretty_generic(loops, net)


@benchmark("VGGNet16")
def perf_train(loops, epochs, net, reader):
    return perf_train_generic(loops, epochs, net, reader)


@benchmark("VGGNet16")
def perf_evaluate(loops, epochs, net, reader):
    return perf_evaluate_generic(loops, epochs, net, reader)


@benchmark("VGGNet16")
def perf_all(loops, reader, epochs):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyVGGNet16()
        net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
        net.set_conv_layer(1, 12, 24, 24, 12, 1, 1)
        net.set_mp_layer(2, 12, 24, 24, 2, 2)

        net.set_conv_layer(3, 12, 12, 12, 24, 3, 3)
        net.set_conv_layer(4, 24, 10, 10, 24, 1, 1)
        net.set_mp_layer(5, 24, 10, 10, 2, 2)

        net.set_conv_layer(6, 24, 5, 5, 24, 1, 1)
        net.set_conv_layer(7, 24, 5, 5, 24, 1, 1)
        net.set_conv_layer(8, 24, 5, 5, 24, 2, 2)
        net.set_mp_layer(9, 24, 4, 4, 2, 2)

        net.set_conv_layer(10, 24, 4, 4, 24, 1, 1)
        net.set_conv_layer(11, 24, 4, 4, 24, 1, 1)
        net.set_conv_layer(12, 24, 4, 4, 24, 1, 1)
        net.set_mp_layer(13, 24, 4, 4, 2, 2)

        net.set_conv_layer(14, 24, 2, 2, 24, 1, 1)
        net.set_conv_layer(15, 24, 2, 2, 24, 1, 1)
        net.set_conv_layer(16, 24, 2, 2, 24, 1, 1)
        net.set_mp_layer(17, 24, 2, 2, 2, 2)

        net.set_dense_layer(18, 24 * 1 * 1, 16)
        net.set_dense_layer(19, 16, 12)
        net.set_dense_layer(20, 12, 10)

        net.set_learning_rate(0.1)
        net.set_adam_beta1(0.997)
        net.set_adam_beta2(0.997)

        net.display()
        net.fine_tune(reader, epochs)
        net.evaluate(reader)

        end = time.time()
        t.append(end - start)
    return t


# utilities

def get_vggnet16():
    net = dll.PyVGGNet16()
    net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
    net.set_conv_layer(1, 12, 24, 24, 12, 1, 1)
    net.set_mp_layer(2, 12, 24, 24, 2, 2)

    net.set_conv_layer(3, 12, 12, 12, 24, 3, 3)
    net.set_conv_layer(4, 24, 10, 10, 24, 1, 1)
    net.set_mp_layer(5, 24, 10, 10, 2, 2)

    net.set_conv_layer(6, 24, 5, 5, 24, 1, 1)
    net.set_conv_layer(7, 24, 5, 5, 24, 1, 1)
    net.set_conv_layer(8, 24, 5, 5, 24, 2, 2)
    net.set_mp_layer(9, 24, 4, 4, 2, 2)

    net.set_conv_layer(10, 24, 4, 4, 24, 1, 1)
    net.set_conv_layer(11, 24, 4, 4, 24, 1, 1)
    net.set_conv_layer(12, 24, 4, 4, 24, 1, 1)
    net.set_mp_layer(13, 24, 4, 4, 2, 2)

    net.set_conv_layer(14, 24, 2, 2, 24, 1, 1)
    net.set_conv_layer(15, 24, 2, 2, 24, 1, 1)
    net.set_conv_layer(16, 24, 2, 2, 24, 1, 1)
    net.set_mp_layer(17, 24, 2, 2, 2, 2)

    net.set_dense_layer(18, 24 * 1 * 1, 16)
    net.set_dense_layer(19, 16, 12)
    net.set_dense_layer(20, 12, 10)

    net.set_learning_rate(0.1)
    net.set_adam_beta1(0.997)
    net.set_adam_beta2(0.997)
    return net
