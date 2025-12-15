.. meta::
  :description: installing GSplat for ROCm
  :keywords: installation instructions, GSplat, AMD, ROCm, Gaussian splatting

.. _gsplat-on-rocm-installation:

********************************************************************
GSplat on ROCm installation
********************************************************************

System requirements
====================================================================

To use GSplat (Gaussian splatting) `1.5.3b2 <https://github.com/ROCm/gsplat/tree/release/1.5.3b2>`__, you need the following prerequisites:

- **ROCm version:** `6.4.3 <https://repo.radeon.com/rocm/apt/6.4.3/>`__ , `7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__ (recommended)
- **Operating system:** Ubuntu 22.04, 24.04
- **GPU platforms:** AMD Instinct™ MI325X, MI300X
- **PyTorch:** `2.6 <https://github.com/ROCm/pytorch/tree/v2.6.0>`__, `2.8 <https://github.com/ROCm/pytorch/tree/v2.8.0>`__ (ROCm-enabled)
- **Python:** `3.10 <https://www.python.org/downloads/release/python-3100/>`__, `3.12 <https://www.python.org/downloads/release/python-3120/>`__ 

Install GSplat
====================================================================

To install GSplat for ROCm, you have the following options:

* :ref:`using-docker-image-with-pytorch` **(recommended)**
* :ref:`build-from-source`

.. _using-docker-image-with-pytorch:

Use a base PyTorch Docker image and install with pip
--------------------------------------------------------------------

You need to install PyTorch to begin. The easiest way is to use the official ROCm-enabled PyTorch Docker image. 
Docker is the recommended method to set up your environment, as it avoids potential installation issues.

1. Pull a ROCm PyTorch Docker image with a supported configuration (see `Docker
   Hub <https://hub.docker.com/r/rocm/pytorch/tags>`__ to browse
   available images). For example:

   .. tab-set::

      .. tab-item:: ROCm 7.0.0
         :sync: rocm7

         .. tab-set::

            .. tab-item:: Ubuntu 24.04
               :sync: ubuntu-24

               .. code-block:: shell

                  docker pull rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

               See `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0
               <https://hub.docker.com/layers/rocm/pytorch/rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0/images/sha256-f6095568e49a2d2f808188920e45f8270a15b9c3f1a7ee49cadb2420e5cf3543>`__
               on Docker Hub.

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu-22

               .. code-block:: shell

                  docker pull rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0

               See `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0
               <https://hub.docker.com/layers/rocm/pytorch/rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0/images/sha256-3f83c28fa8947bed6d7b127c528d5ed72137f9fda48e67f6e9c2d28191ff7aeb>`__
               on Docker Hub.

      .. tab-item:: ROCm 6.4.3
         :sync: rocm6

         .. tab-set::

            .. tab-item:: Ubuntu 24.04
               :sync: ubuntu-24

               .. code-block:: shell

                  docker pull rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

               See `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
               <https://hub.docker.com/layers/rocm/pytorch/rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0/images/sha256-92fcaa70aad9e1909b9dda90c9614545081bb653b38d79bc98494b9441777e53>`__
               on Docker Hub.

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu-22

               .. code-block:: shell

                  docker pull rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0

               See `rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0
               <https://hub.docker.com/layers/rocm/pytorch/rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0/images/sha256-abf137e10a5f3c47b5dbb1bc4e56cf9d97238b6559f39a084ed61cc77fa6140e>`__
               on Docker Hub.

2. Launch and connect to the Docker container:

   .. tab-set::

      .. tab-item:: ROCm 7.0.0
         :sync: rocm7

         .. tab-set::

            .. tab-item:: Ubuntu 24.04
               :sync: ubuntu-24

               .. code-block:: shell

                  docker run -it \
                      --cap-add=SYS_PTRACE \
                      --ipc=host \
                      --privileged=true \
                      --shm-size=128GB \
                      --network=host \
                      --device=/dev/kfd \
                      --device=/dev/dri \
                      --group-add video \
                      -v $HOME:$HOME \
                      --name rocm_pytorch \
                      rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu-22

               .. code-block:: shell

                  docker run -it \
                      --cap-add=SYS_PTRACE \
                      --ipc=host \
                      --privileged=true \
                      --shm-size=128GB \
                      --network=host \
                      --device=/dev/kfd \
                      --device=/dev/dri \
                      --group-add video \
                      -v $HOME:$HOME \
                      --name rocm_pytorch \
                      rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0

      .. tab-item:: ROCm 6.4.3
         :sync: rocm6

         .. tab-set::

            .. tab-item:: Ubuntu 24.04
               :sync: ubuntu-24

               .. code-block:: shell

                  docker run -it \
                      --cap-add=SYS_PTRACE \
                      --ipc=host \
                      --privileged=true \
                      --shm-size=128GB \
                      --network=host \
                      --device=/dev/kfd \
                      --device=/dev/dri \
                      --group-add video \
                      -v $HOME:$HOME \
                      --name rocm_pytorch \
                      rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

            .. tab-item:: Ubuntu 22.04
               :sync: ubuntu-22

               .. code-block:: shell

                  docker run -it \
                      --cap-add=SYS_PTRACE \
                      --ipc=host \
                      --privileged=true \
                      --shm-size=128GB \
                      --network=host \
                      --device=/dev/kfd \
                      --device=/dev/dri \
                      --group-add video \
                      -v $HOME:$HOME \
                      --name rocm_pytorch \
                      rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0

3. After setting up the container, install GSplat from the AMD-hosted PyPI repository:

   .. tab-set::

      .. tab-item:: ROCm 7.0.0
         :sync: rocm7

         .. code-block:: bash

            pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/

      .. tab-item:: ROCm 6.4.3
         :sync: rocm6

         .. code-block:: bash

            pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/

4. Verify the installation:

   .. code-block:: bash

      pip show amd_gsplat

5. The output should show as follows:

   .. code-block:: text

      Name: amd_gsplat
      Version: 1.5.3+30c1e78
      Summary: Python package for differentiable rasterization of gaussians
      Home-page: https://github.com/rocm/gsplat
      Author: AMD Corporation
.. _build-from-source:

Build GSplat from source
--------------------------------------------------------------------

You can choose to setup the development environment and build the GSplat library manually with the following steps.

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before building GSplat from source, ensure you have one of the following installed:

- `ROCm 7.0.0 <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.0/>`__ and `PyTorch with ROCm 7.0.0 <https://hub.docker.com/r/rocm/pytorch/tags?name=7.0_&ordering=name>`__ 
- `ROCm 6.4.3 <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.3/>`__ and `PyTorch with ROCm 6.4.3 <https://hub.docker.com/r/rocm/pytorch/tags?name=6.4.3&ordering=name>`__ 

1. To verify your setup, check the ROCm installation and GPU architecture:

   .. code-block:: bash

      rocminfo | grep gfx

2. The output should be as follows: 

   .. code-block:: bash

      Name:                    gfx942
            Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
            Name:                    amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-

3. Confirm that PyTorch is installed and accessible:

   .. code-block:: bash

      python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'

4. Confirm that the GPU is accessible from PyTorch:

   .. code-block:: bash

      python3 -c 'import torch; print(torch.cuda.is_available())'

Build steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository with submodules:

   .. code-block:: bash

      git clone --recurse-submodules https://github.com/rocm/gsplat.git

2. Install the GLM dependency:

   .. code-block:: bash

      cd gsplat/cuda/csrc/third_party/glm
      cmake \
          -DGLM_BUILD_TESTS=OFF \
          -DBUILD_SHARED_LIBS=OFF \
          -B build . \
          -DCMAKE_INSTALL_PREFIX:PATH=~/.local
      cmake --build build -- all
      cmake --build build -- install

3. Build the GSplat wheel:

   .. code-block:: bash

      cd ../../../..
      python setup.py bdist_wheel

4. Install the wheel:

   .. code-block:: bash

      pip install dist/amd_gsplat*.whl

5. Verify the installation:

   .. code-block:: bash

      pip show amd_gsplat

Run unit tests and verify the installation
====================================================================

The ``tests/`` folder contains automated test scripts that verify that the GPU implementations match those of PyTorch. 
They primarily focus on validating the functionality of the Gaussian Splatting (GS) and 3D rendering components. 
These tests ensure the correctness, performance, and stability of the core features implemented in the GSplat library.

1. To run the tests, first install the `nerfacc <https://github.com/nerfstudio-project/nerfacc>`__ dependency:

   .. code-block:: bash

      git clone https://github.com/rocm/nerfacc.git
      cd nerfacc
      python setup.py bdist_wheel
      pip install dist/amd_nerfacc*.whl

2. Run specific tests from the ``tests`` directory:

   .. code-block:: bash

      cd tests
      pytest -s -v test_2dgs.py::test_fully_fused_projection_packed_2dgs
      pytest -s -v test_2dgs.py
      pytest -s -v test_basic.py

Run a GSplat example
====================================================================

A set of examples is available to help you get started. See :doc:`Run a GSplat example <../examples/gsplat-examples>` for more details.

Benchmarking and evaluation
====================================================================

The `https://github.com/ROCm/gsplat <https://github.com/ROCm/gsplat>`_ repository includes a standalone script that reproduces the 
official Gaussian Splatting results with matching performance on ``PSNR``, ``SSIM``, ``LPIPS``, and the converged number of Gaussians.
See :doc:`benchmarks <../reference/benchmark-evaluation>` for more details.

Limitations
-------------------------------------------------------------------

- Compression of ``splat parameters`` (``positions``, ``scales``, ``rotations``, ``colors``, and ``features``) using PNG image encoding is not supported.  
- Usage of ``Fused Bilagrid`` in the simple trainer is not supported (torch-based implementations are used instead).  
- Distributed training with the ``packed`` option is not supported. 
