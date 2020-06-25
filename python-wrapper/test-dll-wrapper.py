import dll as dll

net = dll.PyDenseDenseNet()
net.display()
net.setLayerSize(0,28*28,16)
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