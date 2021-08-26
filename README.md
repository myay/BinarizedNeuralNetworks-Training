# BNN-Training

First, install pytorch.

## CPU-based Training

If no GPU is available, just run

```python3 run_fashion_cpu.py --model=cnn --batch-size=256 --epochs=100 --lr=0.001 --step-size=25 --no-cuda --cpu-binarization=1```.


## CUDA-based Training and Binarization

For faster binarization, specialized CUDA kernel support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install CUDA-kernels for fast binarization, go to folder ```code/cuda/binarizationPM1``` and run

```python3 setup.py install --user```

After successful installation, run the GPU-based training with

```python3 run_fashion.py --model=cnn --batch-size=256 --epochs=10 --lr=0.001 --step-size=25```.

A small convolutional neural network (CNN) is used by default. To try the fully connected case, use ```--model=fc``` instead of ```--model=cnn``` in the arguments.

The code is based on the MNIST example in https://github.com/pytorch/examples/tree/master/mnist.
