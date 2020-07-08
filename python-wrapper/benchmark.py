import sys
from os import mkdir, path

import benchmark.common as c
import benchmark.dd_perf as dd
import benchmark.ddd_perf as ddd
import benchmark.lenet_perf as ln
import benchmark.alexnet_perf as an
import benchmark.vggnet19_perf as vgg

# path where to store the files with the output
path_benchmark = "../benchmark"

# Create directory if not existing
try:
    if not path.exists(path_benchmark):
        print("Create missing output directory %s " % path_benchmark)
        mkdir(path_benchmark)
except OSError:
    print("Creation of the directory %s failed" % path_benchmark)

if len(sys.argv) < 2:
    print("No network selected for benchmark...\n"
          "Usage : python benchmark.py net_name...\n"
          "Existing network names:      dd          2x dense layers network\n"
          "                             ddd         3x dense layers network\n"
          "                             lenet       network LeNet5\n"
          "                             alexnet     network AlexNet\n"
          "                             vggnet19    network VGGNnet19\n")

# ========================= 2xDense =========================
if "dd" in sys.argv:
    file = open(path_benchmark + "benchmark_py_dd.txt", "w")
    dd.perf_init1(file, 10000)
    dd.perf_init2(file, 10000)
    dd.perf_display(file, 10000, dd.get_2xdense_net())
    dd.perf_display_pretty(file, 10000, dd.get_2xdense_net())
    dd.perf_train(file, 50, 25, dd.get_2xdense_net(), c.get_mnist_reader())
    dd.perf_evaluate(file, 50, 25, dd.get_2xdense_net(), c.get_mnist_reader())
    dd.perf_all(file, 50, c.get_mnist_reader(), 25)
    file.close()

# ========================= 3xDense =========================
if "ddd" in sys.argv:
    file = open(path_benchmark + "benchmark_py_ddd.txt", "w")
    ddd.perf_init1(file, 10000)
    ddd.perf_init2(file, 10000)
    net = ddd.get_3xdense_net()
    ddd.perf_display(file, 10000, net)
    ddd.perf_display_pretty(file, 10000, ddd.get_3xdense_net())
    ddd.perf_train(file, 50, 25, ddd.get_3xdense_net(), c.get_mnist_reader())
    ddd.perf_evaluate(file, 50, 25, ddd.get_3xdense_net(), c.get_mnist_reader())
    ddd.perf_all(file, 50, c.get_mnist_reader(), 25)
    file.close()

# ========================== LeNet ==========================
if "lenet" in sys.argv:
    file = open(path_benchmark + "benchmark_py_lenet.txt", "w")
    ln.perf_init(10000)
    ln.perf_display(10000, ln.get_lenet())
    ln.perf_display_pretty(10000, ln.get_lenet())
    ln.perf_train(50, 25, ln.get_lenet(), c.get_mnist_reader())
    ln.perf_evaluate(50, 25, ln.get_lenet(), c.get_mnist_reader())
    ln.perf_all(50, c.get_mnist_reader(), 25)
    file.close()

# ========================= AlexNet =========================
if "alexnet" in sys.argv:
    file = open(path_benchmark + "benchmark_alexnet.txt", "w")
    an.perf_init(10000)
    an.perf_display(10000, an.get_alexnet())
    an.perf_display_pretty(10000, an.get_alexnet())
    an.perf_train(10, 15, an.get_alexnet(), c.get_mnist_reader())
    an.perf_evaluate(10, 15, an.get_alexnet(), c.get_mnist_reader())
    an.perf_all(10, c.get_mnist_reader(), 15)
    file.close()

# ========================= VGGNet19 =========================
if "vggnet19" in sys.argv:
    file = open(path_benchmark + "benchmark_py_vggnet19.txt", "w")
    vgg.perf_init(file, 10000)
    vgg.perf_display(file, 10000, vgg.get_vggnet19())
    vgg.perf_display_pretty(file, 10000, vgg.get_vggnet19())
    vgg.perf_train(file, 10, 15, vgg.get_vggnet19(), c.get_mnist_reader())
    vgg.perf_evaluate(file, 10, 15, vgg.get_vggnet19(), c.get_mnist_reader())
    vgg.perf_all(file, 10, c.get_mnist_reader(), 15)
    file.close()
