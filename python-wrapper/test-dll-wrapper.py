import mnist_reader as mnist

# Execute simple example
# dll.do_simple_example()

# Createt instance of PyMnistLib
# mnist_lib = dll.PyMnistLib()
# mnist_lib.create_net(28 * 28, 32, 0.001)
# mnist_lib.display_net()
# mnist_lib.display_dataset()
# mnist_lib.train(5)
# mnist_lib.evaluate()

# Print first sample of training datas
# print(mnist_lib.get_dataset()['training_images'][0])

# Print test labels
reader = mnist.PyMnistReader()
reader.display()
reader.display_pretty()
print(reader.read_dataset()['test_labels'][:10])