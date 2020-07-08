import dll
import time
from benchmark.common import *

@benchmark("AlexNet")
def perf_init(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyAlexNet()
        net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
        net.set_mp_layer(1, 12, 24, 24, 2, 2)
        net.set_conv_layer(2, 12, 12, 12, 12, 3, 3)
        net.set_mp_layer(3, 12, 10, 10, 2, 2)
        net.set_conv_layer(4, 12, 5, 5, 12, 1, 1)
        net.set_conv_layer(5, 12, 5, 5, 12, 1, 1)
        net.set_conv_layer(6, 12, 5, 5, 12, 2, 2)
        net.set_mp_layer(7, 12, 4, 4, 2, 2)

        net.set_dense_layer(8, 12 * 2 * 2, 32)
        net.set_dense_layer(9, 32, 10)

        net.set_learning_rate(0.1)
        net.set_adam_beta1(0.997)
        net.set_adam_beta2(0.997)

        end = time.time()
        t.append(end - start)
    return t


@benchmark("AlexNet")
def perf_display(loops, net):
    return perf_display_generic(loops, net)


@benchmark("AlexNet")
def perf_display_pretty(loops, net):
    return perf_display_pretty_generic(loops, net)


@benchmark("AlexNet")
def perf_train(loops, epochs, net, reader):
    return perf_train_generic(loops, epochs, net, reader)


@benchmark("AlexNet")
def perf_evaluate(loops, epochs, net, reader):
    return perf_evaluate_generic(loops, epochs, net, reader)

@benchmark("AlexNet")
def perf_all(loops, reader, epochs):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyAlexNet()
        net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
        net.set_mp_layer(1, 12, 24, 24, 2, 2)
        net.set_conv_layer(2, 12, 12, 12, 12, 3, 3)
        net.set_mp_layer(3, 12, 10, 10, 2, 2)
        net.set_conv_layer(4, 12, 5, 5, 12, 1, 1)
        net.set_conv_layer(5, 12, 5, 5, 12, 1, 1)
        net.set_conv_layer(6, 12, 5, 5, 12, 2, 2)
        net.set_mp_layer(7, 12, 4, 4, 2, 2)

        net.set_dense_layer(8, 12 * 2 * 2, 32)
        net.set_dense_layer(9, 32, 10)

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

def get_alexnet():
    net = dll.PyAlexNet()
    net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
    net.set_mp_layer(1, 12, 24, 24, 2, 2)
    net.set_conv_layer(2, 12, 12, 12, 12, 3, 3)
    net.set_mp_layer(3, 12, 10, 10, 2, 2)
    net.set_conv_layer(4, 12, 5, 5, 12, 1, 1)
    net.set_conv_layer(5, 12, 5, 5, 12, 1, 1)
    net.set_conv_layer(6, 12, 5, 5, 12, 2, 2)
    net.set_mp_layer(7, 12, 4, 4, 2, 2)
    net.set_dense_layer(8, 12 * 2 * 2, 32)
    net.set_dense_layer(9, 32, 10)

    net.set_learning_rate(0.1)
    net.set_adam_beta1(0.997)
    net.set_adam_beta2(0.997)
    return net

