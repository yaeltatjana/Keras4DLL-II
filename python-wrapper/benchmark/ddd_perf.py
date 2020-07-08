import dll
import time
from benchmark.common import *


@benchmark("3xdense_net")
def perf_init1(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyDenseDenseDenseNet([28 * 28, 32, 16], [32, 16, 10])
        net.set_initial_momentum(0.85)

        end = time.time()
        t.append(end - start)
    return t


@benchmark("3xdense_net")
def perf_init2(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyDenseDenseDenseNet()
        net.set_layer_size(0, 28 * 28, 32)
        net.set_layer_size(1, 32, 16)
        net.set_layer_size(2, 16, 10)
        net.set_initial_momentum(0.85)

        end = time.time()
        t.append(end - start)

    return t


@benchmark("3xdense_net")
def perf_display(loops, net):
    return perf_display_generic(loops, net)


@benchmark("3xdense_net")
def perf_display_pretty(loops, net):
    return perf_display_pretty_generic(loops, net)


@benchmark("3xdense_net")
def perf_train(loops, epochs, net, reader):
    return perf_train_generic(loops, epochs, net, reader)


@benchmark("3xdense_net")
def perf_evaluate(loops, epochs, net, reader):
    return perf_evaluate_generic(loops, epochs, net, reader)


@benchmark("3xdense_net")
def perf_all(loops, reader, epochs):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyDenseDenseDenseNet()
        net.set_layer_size(0, 28 * 28, 32)
        net.set_layer_size(1, 32, 16)
        net.set_layer_size(2, 16, 10)
        net.set_initial_momentum(0.85)

        net.display()
        net.fine_tune(reader, epochs)
        net.evaluate(reader)

        end = time.time()
        t.append(end - start)
    return t


def get_3xdense_net():
    net = dll.PyDenseDenseDenseNet()
    net.set_layer_size(0, 28 * 28, 32)
    net.set_layer_size(1, 32, 16)
    net.set_layer_size(2, 16, 10)
    net.set_initial_momentum(0.85)
    return net
