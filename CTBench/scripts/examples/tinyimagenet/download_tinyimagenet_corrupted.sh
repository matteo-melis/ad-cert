#!/bin/bash

mkdir -p ./data
cd ./data

wget https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar
tar -xvf Tiny-ImageNet-C.tar
rm Tiny-ImageNet-C.tar

cd ..