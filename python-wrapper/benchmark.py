import benchmark.lenet_perf as ln
import benchmark.alexnet_perf as an
import benchmark.vggnet19_perf as vgg
import benchmark.ddd_perf as ddd
import benchmark.dd_perf as dd
import benchmark.common as c

# TODO: add text file for the two first

# ========================== LeNet ==========================
# ln.perf_init(1000)
# ln.perf_display(1000, ln.get_lenet())
# ln.perf_display_pretty(1000, ln.get_lenet())
# ln.perf_train(2, 2, ln.get_lenet(), c.get_mnist_reader())
# ln.perf_evaluate(10, ln.get_lenet(), c.get_mnist_reader())
# ln.perf_all(2, c.get_mnist_reader(), 5)

# ========================= AlexNet =========================
# an.perf_init(1000)
# an.perf_display(1000, an.get_alexnet())
# an.perf_display_pretty(1000, an.get_alexnet())
# an.perf_train(2, 2, an.get_alexnet(), c.get_mnist_reader())
# an.perf_evaluate(10, an.get_alexnet(), c.get_mnist_reader())
# an.perf_all(2, c.get_mnist_reader(), 5)

# ========================= 3xDense =========================
# file = open("out/benchmark_ddd.txt", "w")  # w for overwrite, a for append in file
# ddd.perf_init1(file, 1000)
# ddd.perf_init2(file, 1000)
# ddd.perf_display(file, 1000, ddd.get_3xdense_net())
# ddd.perf_display_pretty(file, 1000, ddd.get_3xdense_net())
# ddd.perf_train(file, 20, 5, ddd.get_3xdense_net(), c.get_mnist_reader())
# ddd.perf_evaluate(file, 1000, ddd.get_3xdense_net(), c.get_mnist_reader())
# ddd.perf_all(file, 20, c.get_mnist_reader(), 5)
# file.close()

# ========================= 2xDense =========================
# file = open("out/benchmark_dd.txt", "w")  # w for overwrite, a for append in file
# dd.perf_init1(file, 1000)
# dd.perf_init2(file, 1000)
# dd.perf_display(file, 1000, dd.get_2xdense_net())
# dd.perf_display_pretty(file, 1000, dd.get_2xdense_net())
# dd.perf_train(file, 20, 5, dd.get_2xdense_net(), c.get_mnist_reader())
# dd.perf_evaluate(file, 1000, dd.get_2xdense_net(), c.get_mnist_reader())
# dd.perf_all(file, 20, c.get_mnist_reader(), 5)
# file.close()

# ========================= 2xDense =========================
file = open("out/benchmark_vggnet19.txt", "w")  # w for overwrite, a for append in file

vgg.perf_init(file, 1000)
vgg.perf_display(file, 1000, vgg.get_vggnet19())
vgg.perf_display_pretty(file, 1000, vgg.get_vggnet19())
vgg.perf_train(file, 2, 2, vgg.get_vggnet19(), c.get_mnist_reader())
vgg.perf_evaluate(file, 2, vgg.get_vggnet19(), c.get_mnist_reader())
vgg.perf_all(file, 2, c.get_mnist_reader(), 2)

file.close()
