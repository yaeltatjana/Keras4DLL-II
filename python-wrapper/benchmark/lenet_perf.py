import dll
import time
from benchmark.common import *


@benchmark("LeNet")
def perf_init(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyLeNet()
        net.set_conv_layer(0, 1, 28, 28, 6, 5, 5)
        net.set_mp_layer(1, 6, 24, 24, 2, 2)
        net.set_conv_layer(2, 6, 12, 12, 16, 5, 5)
        net.set_mp_layer(3, 16, 8, 8, 2, 2)
        net.set_dense_layer(4, 4 * 4 * 16, 150)
        net.set_dense_layer(5, 150, 10)

        net.set_learning_rate(0.1)
        net.set_adam_beta1(0.997)
        net.set_adam_beta2(0.997)

        end = time.time()
        t.append(end - start)
    return t


@benchmark("LeNet")
def perf_display(loops, net):
    return perf_display_generic(loops, net)


@benchmark("LeNet")
def perf_display_pretty(loops, net):
    return perf_display_pretty_generic(loops, net)


@benchmark("LeNet")
def perf_train(loops, epochs, net, reader):
    return perf_train_generic(loops, epochs, net, reader)


@benchmark("LeNet")
def perf_evaluate(loops, net, reader):
    return perf_evaluate_generic(loops, net, reader)


@benchmark("LeNet")
def perf_all(loops, reader, epochs):
    t = []
    for x in range(0, loops):
        start = time.time()
        net = dll.PyLeNet()
        net.set_conv_layer(0, 1, 28, 28, 6, 5, 5)
        net.set_mp_layer(1, 6, 24, 24, 2, 2)
        net.set_conv_layer(2, 6, 12, 12, 16, 5, 5)
        net.set_mp_layer(3, 16, 8, 8, 2, 2)
        net.set_dense_layer(4, 4 * 4 * 16, 150)
        net.set_dense_layer(5, 150, 10)
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

def get_lenet():
    net = dll.PyLeNet()
    net.set_conv_layer(0, 1, 28, 28, 6, 5, 5)
    net.set_mp_layer(1, 6, 24, 24, 2, 2)
    net.set_conv_layer(2, 6, 12, 12, 16, 5, 5)
    net.set_mp_layer(3, 16, 8, 8, 2, 2)
    net.set_dense_layer(4, 4 * 4 * 16, 150)
    net.set_dense_layer(5, 150, 10)
    net.set_learning_rate(0.1)
    net.set_adam_beta1(0.997)
    net.set_adam_beta2(0.997)
    return net
