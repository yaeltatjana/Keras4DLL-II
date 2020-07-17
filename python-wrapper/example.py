import dll as dll

if __name__ == "__main__":
    # MNIST dataset
    mnist = dll.PyMnistReader()
    mnist.display()

    # Network with 2 dense layers: relu -> softmax
    net = dll.PyDenseDenseNet([28 * 28, 16], [16, 10])
    net.display()
    net.set_initial_momentum(0.8)

    net.fine_tune(mnist, 5)
    net.evaluate(mnist)


