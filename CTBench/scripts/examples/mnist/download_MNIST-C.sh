#!/bin/bash

mkdir -p ./data
cd ./data

wget https://zenodo.org/records/3239543/files/mnist_c.zip
unzip mnist_c.zip
rm mnist_c.zip

cd ..
