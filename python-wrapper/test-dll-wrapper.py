import wrapperlibmnist as dll

# Execute simple example
dll.do_simple_example()

# Createt instance of PyMnistLib
mnist_lib = dll.PyMnistLib()

# Print first sample of training datas
print(mnist_lib.get_dataset()['training_images'][0])

# Print test labels
print(mnist_lib.get_dataset()['test_labels'])
