# GSplat for ROCm

**GSplat** is an open-source library for GPU-accelerated rasterization of Gaussians with Python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

This repository is the HIP port of the original `GSplat` project, optimized for **ROCm**, and designed to run on AMD Instinct™ GPUs. 

## System Requirements

To use GSplat, you need the following prerequisites:

- **ROCm**: version 6.4.3, 7.0.0 (recommended)
- **Operating system**: Ubuntu 22.04, 24.04  
- **GPU platform**: AMD Instinct™ MI300X  
- **PyTorch**: version 2.6, 2.8 (ROCm-enabled)  
- **Python**: version 3.10, 3.12  

## Installation

1. Install PyTorch (with ROCm support).  
   The easiest method is using the official ROCm PyTorch Docker image:

   For ROCm 7.0.0:

   ```bash
   docker pull rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
   ```

   For ROCm 6.4.3:

   ```bash
   docker pull rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0
   ```

2. Launch and connect to the container:

   For ROCm 7.0.0:

   ```bash
   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true      --shm-size=128GB --network=host      --device=/dev/kfd --device=/dev/dri      --group-add video -it -v $HOME:$HOME      --name rocm_pytorch rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
   ```

   For ROCm 6.4.3:

   ```bash
   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true      --shm-size=128GB --network=host      --device=/dev/kfd --device=/dev/dri      --group-add video -it -v $HOME:$HOME      --name rocm_pytorch rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0
   ```

3. Install GSplat from the AMD-hosted PyPI repository:

   For ROCm 7.0.0:

   ```bash
   pip install gsplat --index-url=https://pypi.amd.com/rocm-7.0.0/simple/
   ```

   For ROCm 6.4.3:

   ```bash
   pip install gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/
   ```

4. Verify the installation:

   ```bash
   pip show gsplat
   ```

5. The output should show as follows:

   ```bash
   Name: gsplat
   Version: 1.5.3+fec758f
   Summary: Python package for differentiable rasterization of Gaussians
   Home-page: https://github.com/rocm/gsplat
   Author: AMD Corporation
   License: Apache 2.0
   Location: /opt/conda/envs/py_3.12/lib/python3.12/site-packages
   Requires: jaxtyping, ninja, numpy, rich, torch


## Examples

We provide a set of examples to get you started. 

1. Clone the examples folder:

   ```bash
   git clone --no-checkout https://github.com/rocm/gsplat.git
   cd gsplat
   git sparse-checkout init --cone
   git sparse-checkout add examples
   git checkout main
   ```

2. Install dependencies and download datasets:

   ```bash
   cd examples
   ./install_dependencies.sh
   python datasets/download_dataset.py
   ```

3. To run the examples, refer to the [run a GSplat example](docs/examples/gsplat-examples.rst) topic. The examples are as follows:

- [Fit a Single Image](docs/examples/gsplat-examples.rst#fit-a-single-image)
- [Fit a 2D image with 3D Gaussians](docs/examples/gsplat-examples.rst#fit-a-single-2d-image-with-3d-gaussians)
- [Render a large scene in real-time](docs/examples/gsplat-examples.rst#render-a-large-scene-in-real-time)

## Evaluation

This repository includes a standalone script that reproduces the official Gaussian Splatting benchmarks with equivalent performance on **PSNR, SSIM, LPIPS**, and the number of converged Gaussians.  

Thanks to GSplat’s optimized GPU implementation:  
- Training uses up to **4× less GPU memory**  
- Training is up to **15% faster** compared to the official implementation  

## Building from source
Refer to the [installation instructions](docs/install/gsplat-install.rst) to learn how to build the GSplat library from source.

## Contributing
We welcome contributions of all kinds and are open to feedback, bug-reports, and improvements, to help expand the capabilities of this software. See [contributing to GSplat](docs/about/contribute-to-gsplat.rst) for more info.

## Core Development

This project is developed and maintained by the following contributors (unordered):  

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley) – Mentor  
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI) – Mentor  
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley) – Project Lead (v0.1)  
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University) – Core Developer  
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley) – Core Developer (v1.0 Lead)  
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley) – Core Developer  
- [Brent Yi](https://github.com/brentyi) (UC Berkeley) – Core Developer  
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University) – Core Developer  
- [Jianbo Ye](http://www.jianboye.org/) (Amazon) – Core Developer  

## Citation

We also provide a white paper with benchmarks, mathematical derivations, and conventions: [arXiv link](https://arxiv.org/abs/2409.06765).  

If you use this library in your research, please cite:

```bibtex
@article{ye2025gsplat,
  title={GSplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```
