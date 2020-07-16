import dll as dll


def test_reader():
    reader = dll.PyMnistReader()
    reader.display()
    reader.display_pretty()
    print(reader.read_dataset()['test_labels'][:10])


def test_3d_reader():
    rd = dll.PyMnist3DReader()
    print(rd.read_training_images()[22])
    print(rd.read_training_labels()[22])


def test_DDNet():
    print("==========================================================================")

    input = [28 * 28, 16]
    output = [16, 10]
    net = dll.PyDenseDenseNet(input, output)
    net.display()
    net.set_initial_momentum(0.85)

    reader2 = dll.PyMnistReader()
    reader2.display()
    net.fine_tune(reader2, 5)
    net.evaluate(reader2)


def test_DDDNet():
    print("==========================================================================")
    input = [28 * 28, 16, 16]
    output = [16, 16, 10]
    net = dll.PyDenseDenseDenseNet(input, output)
    net.display()
    net.set_initial_momentum(0.85)

    reader2 = dll.PyMnistReader()
    reader2.display()
    net.fine_tune(reader2, 5)
    net.evaluate(reader2)
    net.store_weights("stored_ddd.txt")


def test_lenet():
    print("==========================================================================")
    net = dll.PyLeNet()
    net.set_conv_layer(0, 1, 28, 28, 6, 5, 5)
    net.set_mp_layer(1, 6, 24, 24, 2, 2)
    net.set_conv_layer(2, 6, 12, 12, 16, 5, 5)
    net.set_mp_layer(3, 16, 8, 8, 2, 2)
    net.set_dense_layer(4, 4 * 4 * 16, 150)
    net.set_dense_layer(5, 150, 10)

    net.set_adam_beta2(0.997)

    net.display()
    r = dll.PyMnistReader()
    net.fine_tune(r, 2)
    net.evaluate(r)


def test_alexnet():
    print("==========================================================================")
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

    net.set_adam_beta2(0.997)

    net.display()
    r = dll.PyMnistReader()
    net.fine_tune(r, 5)
    net.evaluate(r)


# TODO: change values, random ones actually
def test_vggnet16():
    print("==========================================================================")
    net = dll.PyVGGNet16()
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

    r = dll.PyMnistReader()
    net.fine_tune(r, 5)
    net.evaluate(r)


def test_text_reader():
    reader = dll.PyTextReader("../dll/test/text_db/images", "../dll/test/text_db/labels")
    print(type(reader.read_images()))

    input = [28 * 28, 16]
    output = [16, 10]
    net = dll.PyDenseDenseNet(input, output)
    net.display()
    net.set_initial_momentum(0.85)

    net.fine_tune(reader, 5)
    net.evaluate(reader)


if __name__ == "__main__":
    test_reader()
    # test_3d_reader()
    # test_DDNet()
    # test_DDDNet()
    # test_lenet()
    # test_alexnet()
    # test_vggnet16()
    # test_text_reader()
