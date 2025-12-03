.. meta::
  :description: GSplat 3DGUT extension
  :keywords: GSplat, ROCm, example, sample, tutorial, how to

.. _use-3dgut-to-extend-gsplat:

********************************************************************
Use 3DGUT to extend GSplat
********************************************************************

`NVIDIA™ 3DGUT <https://research.nvidia.com/labs/toronto-ai/3DGUT/>`__ is now integrated into GSplat, 
which extends 3D Gaussian Splatting (3DGS) to support nonlinear camera projections such as distortions in pinhole or fisheye cameras, 
and `rolling shutter <https://en.wikipedia.org/wiki/Rolling_shutter>`__ effects. 
This lets you directly train 3DGS on captured images without the need of undistorting them beforehand. 
However, camera calibration (e.g., using COLMAP) is still required to get distortion parameters).

Directly run an example
====================================================================

To enable training with 3DGUT when running ``examples`` in GSplat, simply pass the following arguments to ``simple_trainer.py``:

``--with_ut --with_eval3d``

Example:

.. code-block:: bash

    python examples/simple_trainer.py mcmc --with_ut --with_eval3d ... <OTHER ARGS>

.. note::

    In GSplat, only the MCMC densification strategy is supported for 3DGUT.

For benchmarking on the MipNeRF360 dataset, see:

``examples/benchmarks/3dgut/mcmc.sh``

If you are not familiar with ``simple_trainer.py``, refer to the ``README.md`` first.

Rendering
====================================================================

Once training is complete, you can view the 3D Gaussian splatting (3DGS) and explore distortion effects supported by 3DGUT using our viewer:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python simple_viewer_3dgut.py --ckpt results/benchmark_mcmc_1M_3dgut/garden/ckpt_29999_rank0.pt

Alternatively, you can use a more comprehensive nerfstudio-style viewer to export videos. Note that changing distortion is **not yet supported** in this viewer:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py --with_ut --with_eval3d --ckpt results/benchmark_mcmc_1M_3dgut/garden/ckpt_29999_rank0.pt

Using the GSplat API
====================================================================

To use 3DGUT through the :doc:`../reference/gsplat-api-reference`, set the following arguments in the ``rasterization()`` function:

- ``with_ut=True`` and ``with_eval3d=True``: Enables 3DGUT. This includes:
    
  - Using the unscented transform to estimate camera projection.
  - Evaluating Gaussian response in 3D space.

- **Pinhole camera with distortion**:
  - Set distortion parameters: ``radial_coeffs``, ``tangential_coeffs``, ``thin_prism_coeffs``.

- **Fisheye camera with distortion**:
  - Set distortion parameters: ``radial_coeffs``
  - Set ``camera_model="fisheye"``

- **F-theta camera with distortion**:
  - Set distortion parameters: ``ftheta_coeffs``
  - Set ``camera_model="ftheta"``

- **Rolling shutter effects**:
  - Check ``rolling_shutter`` and ``viewmats_rs`` for the supported types of rolling shutters.

