# gsplat for ROCm

gsplat is an open-source library for GPU  accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

This repo is the HIP port of the gsplat repo which is enabled and optimised for RCOM runing on AMD Instinct processors. 

## System Requirements
GSPLAT  for AMD ROCm™ 1.5.3 depends directly on NumPy and PyTorch for AMD ROCm™ with many optional dependencies

- OS : Ubuntu 22.04 and Ubuntu 24.04
- ROCM Versions 6.4 and 7.0
- GPU Platforms : MI300, MI325, and MI350
- Pytorch: 2.6, 2.7, and 2.8
- Python : 3.12 and above.
  
## Installation

 - Install [Pytorch](https://pytorch.org/get-started/locally/) first. The easiet way is the use the official docker image which has pytorch for ROCm.

   Rocm 6.4
    ```bash
    docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true   --shm-size=128GB --network=host --device=/dev/kfd  --device=/dev/dri --group-add video -it   -v $HOME:$HOME --name rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0
 
    ```

   Rocm 7.0
   ```bash
   docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true   --shm-size=128GB --network=host --device=/dev/kfd  --device=/dev/dri --group-add video -it   -v $HOME:$HOME --name rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
   ```
 
- Install the gsplat from the AMD hosted PYPI repoitory.
  
  Rocm 6.4
    ```bash
    pip install gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/
    ```

  Rocm 7.0
    ```bash
    pip install gsplat --index-url=https://pypi.amd.com/rocm-7.0.0/simple/
    ```
    
- Once the installation is sucessful, you can verify the installation using the pip show command
  ```bash
    pip show gsplat
  ```

## Examples

We provide a set of examples to get you started! Below you can find the details about the examples 

- Get the examples folder from the github repo
  ```bash
  git clone --no-checkout https://github.com/rocm/gsplat.git && cd gsplat && git sparse-checkout init --cone
  git sparse-checkout add examples && git checkout main
  ```
- Install the dependencies and download the test data set
   ```bash 
   cd examples
   ./install_dependencies.sh
   python datasets/download_dataset.py
  ```
- To run the examples refer to the docs below.
  - [Fit a Single Image](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1128163102/Fit+a+single+image)
  
  - [Fit a 2D image with 3D Gaussians.](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1128134447/Fit+a+COLMAP+Capture)
  
  - [Render a large scene in real-time.](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1128302047/Render+a+Large+Scene)

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient GPU implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1115965044/Evaluation+Results+on+MI300).

## Building from source
Please checkout our [dev-setup](docs/DevSetup.md) documentation for instructions on how to build the gsplat library from source.

## Contributing
We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [Contributing.md](docs/Contributing.md) for more info.


## Core Development

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```
