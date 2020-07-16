import sys
from os import mkdir, path

import benchmark.common as c
import benchmark.dd_perf as dd
import benchmark.ddd_perf as ddd
import benchmark.lenet_perf as ln
import benchmark.alexnet_perf as an
import benchmark.vggnet16_perf as vgg

# path where to store the files with the output
path_benchmark = path.dirname(path.abspath(__file__)) + "/../benchmark/"
prefix_file = "benchmark_py_"

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
          "                             vggnet16    network VGGNnet16\n")

# ========================= 2xDense =========================
if "dd" in sys.argv:
    file = open(path_benchmark + prefix_file + "dd.txt", "w")
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
    file = open(path_benchmark + prefix_file + "ddd.txt", "w")
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
    file = open(path_benchmark + prefix_file + "lenet.txt", "w")
    ln.perf_init(file, 10000)
    ln.perf_display(file, 10000, ln.get_lenet())
    ln.perf_display_pretty(file, 10000, ln.get_lenet())
    ln.perf_train(file, 50, 25, ln.get_lenet(), c.get_mnist_reader())
    ln.perf_evaluate(file, 50, 25, ln.get_lenet(), c.get_mnist_reader())
    ln.perf_all(file, 50, c.get_mnist_reader(), 25)
    file.close()

# ========================= AlexNet =========================
if "alexnet" in sys.argv:
    file = open(path_benchmark + prefix_file + "alexnet.txt", "w")
    an.perf_init(file, 10000)
    an.perf_display(file, 10000, an.get_alexnet())
    an.perf_display_pretty(file, 10000, an.get_alexnet())
    an.perf_train(file, 10, 15, an.get_alexnet(), c.get_mnist_reader())
    an.perf_evaluate(file, 10, 15, an.get_alexnet(), c.get_mnist_reader())
    an.perf_all(file, 10, c.get_mnist_reader(), 15)
    file.close()

# ========================= VGGNet16 =========================
if "vggnet16" in sys.argv:
    file = open(path_benchmark + prefix_file + "vggnet16.txt", "w")
    vgg.perf_init(file, 10000)
    vgg.perf_display(file, 10000, vgg.get_vggnet16())
    vgg.perf_display_pretty(file, 10000, vgg.get_vggnet16())
    vgg.perf_train(file, 10, 15, vgg.get_vggnet16(), c.get_mnist_reader())
    vgg.perf_evaluate(file, 10, 15, vgg.get_vggnet16(), c.get_mnist_reader())
    vgg.perf_all(file, 10, c.get_mnist_reader(), 15)
    file.close()
