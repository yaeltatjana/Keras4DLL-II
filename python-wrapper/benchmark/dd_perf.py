import dll
import time
from benchmark.common import *


@benchmark("2xdense_net")
def perf_init1(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyDenseDenseNet([28 * 28, 16], [16, 10])
        net.set_initial_momentum(0.85)

        end = time.time()
        t.append(end - start)
    return t


@benchmark("2xdense_net")
def perf_init2(loops):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyDenseDenseNet()
        net.display()
        net.set_layer_size(0, 28 * 28, 16)
        net.set_layer_size(1, 16, 10)
        net.set_initial_momentum(0.85)

        end = time.time()
        t.append(end - start)

    return t


@benchmark("2xdense_net")
def perf_display(loops, net):
    return perf_display_generic(loops, net)


@benchmark("2xdense_net")
def perf_display_pretty(loops, net):
    return perf_display_pretty_generic(loops, net)


@benchmark("2xdense_net")
def perf_train(loops, epochs, net, reader):
    return perf_train_generic(loops, epochs, net, reader)


@benchmark("2xdense_net")
def perf_evaluate(loops, net, reader):
    return perf_evaluate_generic(loops, net, reader)


@benchmark("2xdense_net")
def perf_all(loops, reader, epochs):
    t = []
    for x in range(0, loops):
        start = time.time()

        net = dll.PyDenseDenseNet()
        net.display()
        net.set_layer_size(0, 28 * 28, 16)
        net.set_layer_size(1, 16, 10)
        net.set_initial_momentum(0.85)

        net.display()
        net.fine_tune(reader, epochs)
        net.evaluate(reader)

        end = time.time()
        t.append(end - start)
    return t


def get_2xdense_net():
    net = dll.PyDenseDenseNet()
    net.set_layer_size(0, 28 * 28, 16)
    net.set_layer_size(1, 16, 10)
    net.set_initial_momentum(0.85)
    return net
