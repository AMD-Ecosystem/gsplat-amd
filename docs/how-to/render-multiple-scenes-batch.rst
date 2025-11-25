.. meta::
  :description: batch render multiple scenes
  :keywords: GSplat, ROCm, example, sample, tutorial, how to

.. _render-multiple-scenes-in-a-batch:

********************************************************************
Render multiple scenes in a batch
********************************************************************

The ``rasterization()`` function now supports arbitrary batching.  
This allows you to render multiple scenes simultaneously.  

For example, you can render **16 different scenes**, each with **6 viewpoints**, in a single call:

.. code-block:: python

    # 16 different scenes, each with 10k Gaussians.
    means: Tensor      # (16, 10000, 3)
    quats: Tensor      # (16, 10000, 4)
    scales: Tensor     # (16, 10000, 3)
    colors: Tensor     # (16, 10000, 3)
    opacities: Tensor  # (16, 10000)

    # Each scene renders to 6 viewpoints (not shared).
    viewmats: Tensor   # (16, 6, 4, 4)
    Ks: Tensor         # (16, 6, 4, 4)

    width, height = 300, 200

    # Render
    # Output `renders` has shape [16, 6, 200, 300, 3]
    # Output `alphas` has shape [16, 6, 200, 300, 1]
    renders, alphas, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height
    )

.. note::

    The API assumes all scenes in a batch have the same number of Gaussians.  
    If your scenes vary in size, you can:
    
    - Pad smaller scenes with zero-opacity Gaussians, or
    - Render them sequentially with a for-loop.

Benchmarking
--------------------------------------------------------

The batching feature was benchmarked with **10,000 Gaussians per scene** (see ``profiling/batch.py``):

+--------+-------------+-----------+---------------+---------------+
| Model  | N Batches   | Mem (GB)  | Time [fwd]    | Time [bwd]    |
+========+=============+===========+===============+===============+
| 3DGS   | 1           | 0.010     | 0.00037       | 0.00049       |
+--------+-------------+-----------+---------------+---------------+
| 3DGS   | 4           | 0.040     | 0.00040       | 0.00079       |
+--------+-------------+-----------+---------------+---------------+
| 3DGS   | 16          | 0.161     | 0.00093       | 0.00284       |
+--------+-------------+-----------+---------------+---------------+
| 3DGS   | 64          | 0.642     | 0.00368       | 0.01124       |
+--------+-------------+-----------+---------------+---------------+
| 3DGUT  | 1           | 0.010     | 0.00042       | 0.00070       |
+--------+-------------+-----------+---------------+---------------+
| 3DGUT  | 4           | 0.040     | 0.00057       | 0.00128       |
+--------+-------------+-----------+---------------+---------------+
| 3DGUT  | 16          | 0.162     | 0.00162       | 0.00513       |
+--------+-------------+-----------+---------------+---------------+
| 3DGUT  | 64          | 0.641     | 0.00635       | 0.02031       |
+--------+-------------+-----------+---------------+---------------+

