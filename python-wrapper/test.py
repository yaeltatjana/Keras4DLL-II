import sys
from os import mkdir, path
import test.dataset_readers as readers
import test.networks as nets
from test.dataset_readers import *
from test.networks import *
import unittest


# creates output directory in the test directory
path_test = path.dirname(path.abspath(__file__)) + "/out/"

# Create directory if not existing
try:
    if not path.exists(path_test):
        print("Create missing output directory %s " % path_test)
        mkdir(path_test)
except OSError:
    print("Creation of the directory %s failed" % path_test)

if len(sys.argv) < 2:
    print("No test selected...\n"
          "Usage : python test.py test_name...\n"
          "Existing tests' names:       dd          2x dense layers network\n"
          "                             ddd         3x dense layers network\n"
          "                             lenet       network LeNet5\n"
          "                             alexnet     network AlexNet\n"
          "                             vggnet19    network VGGNnet19\n"
          "                             mnist       MNIST dataset reader\n"
          "                             text        text dataset reader\n")


suite = unittest.TestSuite()
readers = ["mnist", "text"]
nets = ["dd","ddd","lenet","alexnet","vggnet19"]

for reader in readers:
    if reader in sys.argv:
        name = "test_" + reader
        print("================ [ " + reader + " ]================\n")

        m = globals()[name]()
        suite.addTest(TestReaders(name))
        # func = getattr(m, name)

for net in nets:
    if net in sys.argv:
        name = "test_" + net
        print("================ [ " + net + " ]================\n")
        m = globals()[name]()
        suite.addTest(TestNets(name))

#
# if "mnist" in sys.argv:
#     readers.test_mnist()
#     sys.argv.remove("mnist")
#     suite.addTest(readers.TestReaders('test_mnist'))
#
#
# if "text" in sys.argv:
#     # nothing to be printed
#     sys.argv.remove("text")
#     suite.addTest(TestReaders('test_text'))
#
#
# if "dd" in sys.argv:
#     nets.test_dd()
#     sys.argv.remove("dd")
#     suite.addTest(TestNets('test_dd'))
#
#
# if "ddd" in sys.argv:
#     nets.test_ddd()
#     sys.argv.remove("ddd")
#     suite.addTest(TestNets('test_ddd'))
#
# if "lenet" in sys.argv:
#     nets.test_ddd()
#     sys.argv.remove("lenet")
#     suite.addTest(TestNets('lenet_ddd'))
#
# if "alexnet" in sys.argv:
#     nets.test_ddd()
#     sys.argv.remove("alexnet")
#     suite.addTest(TestNets('alexnet_ddd'))
#
# if "vggnet19" in sys.argv:
#     nets.test_ddd()
#     sys.argv.remove("vggnet19")
#     suite.addTest(TestNets('vggnet19_ddd'))


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
