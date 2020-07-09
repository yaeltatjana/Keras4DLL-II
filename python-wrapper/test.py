import sys
from os import mkdir, path
from test.dataset_readers import *
from test.networks import *
import unittest

# creates output directory in the test directory
path_test = path.dirname(path.abspath(__file__)) + "/test/out/"
print(path.dirname(path.abspath(__file__)) + "/test/out/")

# Create directory if not existing
try:
    if not path.exists(path_test):
        print("Create missing output directory %s " % path_test)
        mkdir(path_test)
except OSError:
    print("Creation of the directory %s failed" % path_test)

# usage of script
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

# suite to add unittests
suite = unittest.TestSuite()

# existing dataset readers
readers = ["mnist", "text"]

# existing nets
nets = ["dd", "ddd", "lenet", "alexnet", "vggnet19"]

arg = sys.argv[1]

if arg in readers:
    name = "test_" + arg
    suite.addTest(TestReaders(name))


if arg in nets:
    name = "test_" + arg
    suite.addTest(TestNets(name))


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)  # to see which unittest is executed
    runner.run(suite)
