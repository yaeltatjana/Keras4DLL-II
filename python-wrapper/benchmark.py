import benchmark.lenet_perf as ln
import benchmark.alexnet_perf as an
import benchmark.common as c

file = open("benchmark.txt", "w")  # w for overwrite, a for append in file

# ========================== LeNet ==========================
ln.perf_init(1000)
ln.perf_display(1000, ln.get_lenet())
ln.perf_display_pretty(1000, ln.get_lenet())
ln.perf_train(2, 2, ln.get_lenet(), c.get_mnist_reader())
ln.perf_evaluate(10, ln.get_lenet(), c.get_mnist_reader())
ln.perf_all(2, c.get_mnist_reader(), 5)

# ========================= AlexNet =========================
an.perf_init(1000)
an.perf_display(1000, an.get_alexnet())
an.perf_display_pretty(1000, an.get_alexnet())
an.perf_train(2, 2, an.get_alexnet(), c.get_mnist_reader())
an.perf_evaluate(10, an.get_alexnet(), c.get_mnist_reader())
an.perf_all(2, c.get_mnist_reader(), 5)
