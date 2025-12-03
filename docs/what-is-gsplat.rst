.. meta::
  :description: What is GSplat?
  :keywords: gaussian splatting, gsplat, AMD, ROCm, overview, introduction

.. _what-is-gsplat:

********************************************************************
What is GSplat?
********************************************************************

`GSplat <https://docs.gsplat.studio/main/>`__ (Gaussian splatting) is an open-source library for GPU-accelerated, differentiable rasterization of 3D Gaussians, with Python bindings. 
It is based on the SIGGRAPH paper `3D Gaussian Splatting for Real-Time Rendering of Radiance Fields <https://doi.org/10.1145/3588432.3591490>`__.
GSplat provides an optimized implementation with improved performance, reduced memory usage, and faster convergence. 
The library features a modular API with Python support, suitable for both research and practical applications.

Gaussian splatting overview
====================================================================

3D Gaussian Splatting (3DGS) is a point-based volumetric method for representing and rendering 3D scenes. It models a scene as a sparse set of anisotropic 3D Gaussian primitives. 

Each primitive has the following:

- Mean position that sets the location
- Covariance matrix that defines shape and orientation
- Color modeled with spherical harmonics
- Opacity that controls contribution to the rendered image

You render a view by projecting the 3D Gaussians onto the image plane and blending their contributions. For each pixel, you accumulate color along the viewing ray from the Gaussians that overlap that pixel. Opacity acts as a weight during accumulation.

You can initialize the Gaussians from 3D points obtained by random sampling or from sources such as `COLMAP <https://colmap.github.io/>`__ or `LiDAR <https://arxiv.org/abs/2409.12899>`__.

.. image:: /images/gaussian_splatting_overview_image.png
   :alt: Diagram of the 3D Gaussian splatting mapping and rendering process

During training, you optimize the parameters of the Gaussians to match the input images. 

A common approach uses a composite loss that combines the following:

- L1 loss (mean absolute error): pixel-level accuracy
- Structural Similarity Index Measure (SSIM): perceptual quality and structural consistency

For optimization, the steps are defined as follows:

- Densification - guided by gradients of projected 2D Gaussian positions; adds Gaussians in sparse or inconsistent regions by splitting larger Gaussians or sampling new ones
- Culling - triggered by low opacity; removes low-contribution Gaussians to improve efficiency while maintaining visual quality

In summary, 3DGS provides an explicit rendering pipeline with a flexible volumetric representation that captures view-dependent appearance, while balancing visual quality and computational cost.

Features and use cases
====================================================================

**Real-time rendering**: Gaussian splatting renders high-quality 3D scenes interactively by representing the scene with a discrete set of 3D Gaussians.  
The rendering process becomes a parallelizable and efficient splatting operation, well-suited for GPU architectures.  
This speed supports applications where frame rate is critical.

**High fidelity and quality**: Gaussian splatting can produce detailed scenes.  
The 3D Gaussians capture geometric and photometric details, including textures, reflections, and lighting effects, resulting in realistic outputs.

**Explicit scene representation**: Unlike NeRFs, Gaussian splatting uses an explicit, discrete set of primitives.  
This makes the representation interpretable and editable.  
You can add, remove, or modify Gaussians, which provides flexibility compared to implicit neural representations.

**Efficient training and optimization**: Creating a GSplat model from images is efficient.  
The optimization process adjusts Gaussian parameters (position, shape, color, opacity) to best match the input.  
Training often takes minutes, compared with hours for NeRF models.

**Compact scene representation**: A GSplat model consists only of Gaussian parameters, making it relatively small in size.  
This compact format simplifies storage, sharing, and deployment, which is especially important for mobile or web delivery.

Performance
====================================================================

GSplat is developed with efficiency in mind:  

- On AMD Instinct™ GPUs, you can leverage greater GPU memory to train larger scenes. See :doc:`benchmarking <reference/benchmark-evaluation>` for more details.
- GSplat supports large scene rendering and performs faster than the official `https://github.com/graphdeco-inria/diff-gaussian-rasterization <https://github.com/graphdeco-inria/diff-gaussian-rasterization>`__.  
- Extra features include ``batch rasterization``, ``N-D feature rendering``, ``depth rendering``, ``sparse gradients``, and ``multi-GPU distributed rasterization``.  
- GSplat integrates techniques such as ``absgrad``, ``anti-aliasing``, and ``3DGS-MCMC``. See the :doc:`API reference <reference/gsplat-api-reference>` for more details. 
