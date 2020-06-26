import dll as dll

def test_reader():
    reader = dll.PyMnistReader()
    reader.display()
    reader.display_pretty()
    print(reader.read_dataset()['test_labels'][:10])

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


test_reader()
test_DDNet()
test_DDDNet()