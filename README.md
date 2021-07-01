# BNN-Training

First, install pytorch. If you want to train on GPU, also install pybind11, CUDA toolkit.


To install CUDA-kernels for binarization, go to folder ```code/cuda/binarizationPM1``` and run

```python3 setup.py install --user```


After successful installation, run the training with

```python3 run_fashion.py --batch-size=256 --epochs=10 --lr=0.001 --step-size=25```.


If no GPU is available, just run

```python3 run_fashion_cpu.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=25 --no-cuda```.


The code is based on the MNIST example in https://github.com/pytorch/examples/tree/master/mnist.
