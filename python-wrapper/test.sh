#!/bin/bash

# you have to be in the dir python-wrapper
# python test.py mnist > test/out/test_mnist_reader.txt
# python test.py text > test/out/test_text_reader.txt
#python test.py dd > test/out/test_dd.txt
#python test.py ddd > test/out/test_ddd.txt
python test.py mnist ddd lenet > test/out/test_all.txt