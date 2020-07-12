#!/bin/bash


# you have to be in the directory python-wrapper
if [ ! -d "test/out" ]
then
  mkdir "test/out"
  echo "Create missing output directory test/out "
fi

python test.py mnist > test/out/test_mnist_reader.txt
# no need to store because there isn't anything
python test.py text # > test/out/test_text_reader.txt
python test.py dd > test/out/test_dd.txt
python test.py ddd > test/out/test_ddd.txt
python test.py lennet > test/out/test_lenet.txt
python test.py alexnet > test/out/test_alexnet.txt
python test.py vggnet16 > test/out/test_vggnet16.txt