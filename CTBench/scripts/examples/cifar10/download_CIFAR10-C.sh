#!/bin/bash

mkdir -p ./data
cd ./data

wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar
rm CIFAR-10-C.tar

cd ..