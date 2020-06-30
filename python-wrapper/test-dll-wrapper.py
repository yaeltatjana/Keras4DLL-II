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
    net = dll.PyDenseDenseNet()
    net.display()
    net.setLayerSize(0,28*28,16)
    net.setLayerSize(1,16,10)
    net.display()
    print("==========================================================================")

    input = [28*28, 16]
    output = [16, 10]
    net2 = dll.PyDenseDenseNet(input, output)
    net2.display()
    net2.setInitialMomentum(0.85)

    reader2 = dll.PyMnistReader()
    reader2.display()
    net2.fineTune(reader2,5)
    net2.evaluate(reader2)


def test_DDDNet():
    print("==========================================================================")
    input = [28*28, 16, 16]
    output = [16, 16, 10]
    net2 = dll.PyDenseDenseDenseNet(input, output)
    net2.display()
    net2.setInitialMomentum(0.85)

    reader2 = dll.PyMnistReader()
    reader2.display()
    net2.fineTune(reader2,5)
    net2.evaluate(reader2)
    net2.store_weights("stored_ddd.txt")

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
    net.fine_tune(r,5)
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
    net.fine_tune(r,5)
    net.evaluate(r)



#test_reader()
#test_3d_reader()
#test_DDNet()
#test_DDDNet()
#test_lenet()
test_alexnet()