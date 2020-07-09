import dll
import unittest

reader = dll.PyMnistReader()


# can only test returned error of training
class TestNets(unittest.TestCase):
    def test_dd(self):
        # first init
        net = dll.PyDenseDenseNet()
        net.set_layer_size(0, 28 * 28, 16)
        net.set_layer_size(1, 16, 10)
        net.set_initial_momentum(0.85)
        net.display()
        net.display_pretty()

        # second init
        net = dll.PyDenseDenseNet([28 * 28, 16], [16, 10])
        net.set_initial_momentum(0.85)
        net.display()
        net.display_pretty()

        net.set_initial_momentum(0.85)
        self.assertLessEqual(net.fine_tune(reader, 5), 0.5)
        net.evaluate(reader)

    def test_ddd(self):
        # first init
        net = dll.PyDenseDenseDenseNet()
        net.set_layer_size(0, 28 * 28, 16)
        net.set_layer_size(1, 16, 16)
        net.set_layer_size(2, 16, 10)
        net.set_initial_momentum(0.85)
        net.display()
        net.display_pretty()

        # second init
        net = dll.PyDenseDenseDenseNet([28 * 28, 16, 16], [16, 16, 10])
        net.set_initial_momentum(0.85)

        net.display()
        net.display_pretty()
        self.assertLessEqual(net.fine_tune(reader, 5), 0.5)
        net.evaluate(reader)

    def test_lenet(self):
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
        net.display_pretty()
        self.assertLessEqual(net.fine_tune(reader, 5), 0.95)
        net.evaluate(reader)

    def test_alexnet(self):
        net = dll.PyAlexNet()
        net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
        net.set_mp_layer(1, 12, 24, 24, 2, 2)
        net.set_conv_layer(2, 12, 12, 12, 12, 3, 3)
        net.set_mp_layer(3, 12, 10, 10, 2, 2)
        net.set_conv_layer(4, 12, 5, 5, 12, 1, 1)
        net.set_conv_layer(5, 12, 5, 5, 12, 1, 1)
        net.set_conv_layer(6, 12, 5, 5, 12, 2, 2)
        net.set_mp_layer(7, 12, 4, 4, 2, 2)

        net.set_dense_layer(8, 12 * 2 * 2, 32)
        net.set_dense_layer(9, 32, 10)

        net.set_learning_rate(0.1)
        net.set_adam_beta1(0.997)
        net.set_adam_beta2(0.997)

        net.display()
        net.display_pretty()
        self.assertLessEqual(net.fine_tune(reader, 5), 0.95)
        net.evaluate(reader)

    def test_vggnet19(self):
        net = dll.PyVGGNet19()
        net.set_conv_layer(0, 1, 28, 28, 12, 5, 5)
        net.set_conv_layer(1, 12, 24, 24, 12, 1, 1)
        net.set_mp_layer(2, 12, 24, 24, 2, 2)

        net.set_conv_layer(3, 12, 12, 12, 24, 3, 3)
        net.set_conv_layer(4, 24, 10, 10, 24, 1, 1)
        net.set_mp_layer(5, 24, 10, 10, 2, 2)

        net.set_conv_layer(6, 24, 5, 5, 24, 1, 1)
        net.set_conv_layer(7, 24, 5, 5, 24, 1, 1)
        net.set_conv_layer(8, 24, 5, 5, 24, 2, 2)
        net.set_mp_layer(9, 24, 4, 4, 2, 2)

        net.set_conv_layer(10, 24, 4, 4, 24, 1, 1)
        net.set_conv_layer(11, 24, 4, 4, 24, 1, 1)
        net.set_conv_layer(12, 24, 4, 4, 24, 1, 1)
        net.set_mp_layer(13, 24, 4, 4, 2, 2)

        net.set_conv_layer(14, 24, 2, 2, 24, 1, 1)
        net.set_conv_layer(15, 24, 2, 2, 24, 1, 1)
        net.set_conv_layer(16, 24, 2, 2, 24, 1, 1)
        net.set_mp_layer(17, 24, 2, 2, 2, 2)

        net.set_dense_layer(18, 24 * 1 * 1, 16)
        net.set_dense_layer(19, 16, 12)
        net.set_dense_layer(20, 12, 10)

        net.set_learning_rate(0.1)
        net.set_adam_beta1(0.997)
        net.set_adam_beta2(0.997)

        net.display()
        net.display_pretty()
        self.assertLessEqual(net.fine_tune(reader, 5), 0.95)
        net.evaluate(reader)
