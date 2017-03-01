#!/bin/bash
# For Time



nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data12.csv cir

sed -i -e 's/12/24/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data24.csv cir

sed -i -e 's/24/36/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data36.csv cir

sed -i -e 's/36/48/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data48.csv cir

sed -i -e 's/48/60/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data60.csv cir

sed -i -e 's/60/72/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data72.csv cir

sed -i -e 's/72/84/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data84.csv cir

sed -i -e 's/84/96/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data96.csv cir

sed -i -e 's/96/108/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data108.csv cir

sed -i -e 's/108/120/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data120.csv cir

sed -i -e 's/120/132/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data132.csv cir

sed -i -e 's/132/144/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data144.csv cir

sed -i -e 's/144/156/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data156.csv cir

sed -i -e 's/156/168/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data168.csv cir

sed -i -e 's/168/180/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data180.csv cir

sed -i -e 's/180/12/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data12.csv vasicek

sed -i -e 's/12/24/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data24.csv vasicek

sed -i -e 's/24/36/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data36.csv vasicek

sed -i -e 's/36/48/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data48.csv vasicek

sed -i -e 's/48/60/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data60.csv vasicek

sed -i -e 's/60/72/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data72.csv vasicek

sed -i -e 's/72/84/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data84.csv vasicek

sed -i -e 's/84/96/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data96.csv vasicek

sed -i -e 's/96/108/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data108.csv vasicek

sed -i -e 's/108/120/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data120.csv vasicek

sed -i -e 's/120/132/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data132.csv vasicek

sed -i -e 's/132/144/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data144.csv vasicek

sed -i -e 's/144/156/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data156.csv vasicek

sed -i -e 's/156/168/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data168.csv vasicek

sed -i -e 's/168/180/g' SpeedTest.cpp
nvcc -std=c++11 -w kernel.cu speedTest.cpp DE.cu
./a.out data180.csv vasicek
