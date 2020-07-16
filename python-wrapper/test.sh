#!/bin/bash

# script to launch the functional tests for the wrapper

# you have to be in the directory python-wrapper
if [ ! -d "test/out" ]
then
  mkdir "test/out"
  echo "Create missing output directory test/out "
fi

python test.py mnist > test/out/test_py_mnist.txt
# no need to store because there isn't anything
python test.py text # > test/out/test_py_text.txt
python test.py dd > test/out/test_py_dd.txt
python test.py ddd > test/out/test_py_ddd.txt
python test.py lenet > test/out/test_py_lenet.txt
python test.py alexnet > test/out/test_py_alexnet.txt
python test.py vggnet16 > test/out/test_py_vggnet162.txt