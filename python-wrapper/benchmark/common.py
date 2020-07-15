import dll
import numpy as np
import time


def benchmark(name):
    def benchmark_func(func):
        def decorate_func(file, loops, *args, **kwargs):
            arr = np.array(func(loops, *args, **kwargs))
            out = "======================= [ {name} - {func} ] =======================\n"
            out += "Average: {avg:.4f} seconds\nMax: {max:.4f} seconds\nMin: {min:.4f} seconds\n"
            out_formatted = out.format(name=name, func=func.__name__, avg=np.sum(arr) / loops, max=np.max(arr), min=np.min(arr))
            if file is None:
                print(out_formatted)
            else:
                file.write(out_formatted)

        return decorate_func

    return benchmark_func


def perf_display_generic(loops, net):
    t = []
    for x in range(0, loops):
        start = time.time()
        net.display()
        end = time.time()
        t.append(end - start)
    return t


def perf_display_pretty_generic(loops, net):
    t = []
    for x in range(0, loops):
        start = time.time()
        net.display_pretty()
        end = time.time()
        t.append(end - start)
    return t


def perf_train_generic(loops, epochs, net, reader):
    t = []
    for x in range(0, loops):
        start = time.time()
        net.fine_tune(reader, epochs)
        end = time.time()
        t.append(end - start)
    return t


def perf_evaluate_generic(loops, epochs, net, reader):
    t = []
    for x in range(0, loops):
        net.fine_tune(reader, epochs)
        start = time.time()
        net.evaluate(reader)
        end = time.time()
        t.append(end - start)
    return t


def get_mnist_reader():
    return dll.PyMnistReader()
