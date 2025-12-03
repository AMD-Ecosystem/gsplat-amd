.. meta::
  :description: Profile the GSplat library
  :keywords: GSplat, ROCm, API, profiling, evaluation 

.. _profiling-gsplat-library:

********************************************************************
Profiling the GSplat library
********************************************************************

This section provides profiling results of GSplat using different rasterization backends and configurations.  
All evaluations were conducted on an AMD Instinct™ MI300X GPU.  

- **Mem:** GPU memory allocated by the forward + backward rasterization process (excluding input data), measured as the difference of ``torch.cuda.max_memory_allocated()`` before and after rasterization.  
- **FPS[fwd/bwd]:** Frames per second for forward/backward passes.  
- **Packed:** Indicates if packed rasterization is enabled.  
- **Sparse Grad:** Indicates if sparse gradient computation is used.  

Render RGB images
====================================================================

**Batch size 1, Channels 3**

.. code-block:: bash

   python profiling/main.py --batch_size 1 --scene_grid 5 --channels 3

+----------------+--------+-------------+------------+------------+------------+
| Backend        | Packed | Sparse Grad | Mem (GB)   | FPS [fwd]  | FPS [bwd]  |
+================+========+=============+============+============+============+
| GSplat v1.5.3  | True   | True        | 0.48       | 376.8      | 209.9      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | True   | False       | 0.48       | 380.2      | 173.7      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | False  | False       | 0.60       | 349.3      | 206.0      |
+----------------+--------+-------------+------------+------------+------------+

**Batch size 4, Channels 3**

.. code-block:: bash

   python profiling/main.py --batch_size 4 --scene_grid 5 --channels 3

+----------------+--------+-------------+------------+------------+------------+
| Backend        | Packed | Sparse Grad | Mem (GB)   | FPS [fwd]  | FPS [bwd]  |
+================+========+=============+============+============+============+
| GSplat v1.5.3  | True   | True        | 1.92       | 143.3      | 102.8      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | True   | False       | 1.92       | 143.6      | 85.4       |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | False  | False       | 2.54       | 144.2      | 89.3       |
+----------------+--------+-------------+------------+------------+------------+

Render feature maps
====================================================================

**Batch size 1, Channels 32**

.. code-block:: bash

   python profiling/main.py --batch_size 1 --scene_grid 1 --channels 32

+----------------+--------+-------------+------------+------------+------------+
| Backend        | Packed | Sparse Grad | Mem (GB)   | FPS [fwd]  | FPS [bwd]  |
+================+========+=============+============+============+============+
| GSplat v1.5.3  | True   | True        | 0.68       | 736.7      | 189.8      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | True   | False       | 0.68       | 740.0      | 184.9      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | False  | False       | 0.67       | 614.1      | 188.6      |
+----------------+--------+-------------+------------+------------+------------+

**Batch size 4, Channels 32**

.. code-block:: bash

   python profiling/main.py --batch_size 4 --scene_grid 1 --channels 32

+----------------+--------+-------------+------------+------------+------------+
| Backend        | Packed | Sparse Grad | Mem (GB)   | FPS [fwd]  | FPS [bwd]  |
+================+========+=============+============+============+============+
| GSplat v1.5.3  | True   | True        | 2.74       | 213.6      | 52.8       |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | True   | False       | 2.74       | 214.3      | 52.3       |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | False  | False       | 2.77       | 217.1      | 52.8       |
+----------------+--------+-------------+------------+------------+------------+

Render a large scene
====================================================================

**49M Gaussians, Batch size 1, Channels 3**

.. code-block:: bash

   python profiling/main.py --batch_size 1 --scene_grid 21 --channels 3

+----------------+--------+-------------+------------+------------+------------+
| Backend        | Packed | Sparse Grad | Mem (GB)   | FPS [fwd]  | FPS [bwd]  |
+================+========+=============+============+============+============+
| GSplat v1.5.3  | True   | True        | 1.38       | 179.0      | 143.8      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | True   | False       | 3.12       | 177.3      | 107.1      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | False  | False       | 5.93       | 179.4      | 93.1       |
+----------------+--------+-------------+------------+------------+------------+

**107M Gaussians, Batch size 1, Channels 3**

.. code-block:: bash

   python profiling/main.py --batch_size 1 --scene_grid 31 --channels 3

+----------------+--------+-------------+------------+------------+------------+
| Backend        | Packed | Sparse Grad | Mem (GB)   | FPS [fwd]  | FPS [bwd]  |
+================+========+=============+============+============+============+
| GSplat v1.5.3  | True   | True        | 2.25       | 135.4      | 138.2      |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | True   | False       | 6.16       | 135.9      | 99.8       |
+----------------+--------+-------------+------------+------------+------------+
| GSplat v1.5.3  | False  | False       | 12.65      | 135.2      | 76.2       |
+----------------+--------+-------------+------------+------------+------------+
