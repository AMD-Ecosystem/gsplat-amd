.. meta::
  :description: Benchmarks for GSplat
  :keywords: benchmarking instructions, GSplat, AMD, ROCm, Gaussian splatting

.. _benchmarks-for-gsplat:

********************************************************************
Benchmarks for GSplat
********************************************************************

The `https://github.com/ROCm/gsplat <https://github.com/ROCm/gsplat>`_ repository includes a standalone script that reproduces the 
official Gaussian Splatting results with matching performance on PSNR, SSIM, LPIPS, and the converged number of Gaussians.

To run the benchmark:

.. code-block:: bash

   cd examples
   ./install_dependencies.sh
   # download mipnerf_360 benchmark data
   python datasets/download_dataset.py
   # run batch evaluation
   bash benchmarks/basic.sh

3D Gaussian Splatting (3DGS) evaluation on AMD Instinct MI300X
====================================================================

The following table summarizes the average evaluation metrics, training memory usage, and training time.

+----------------+--------+-------+-------+----------------+------------+
| Model          | PSNR   | SSIM  | LPIPS | Train Memory   | Train Time |
|                | (dB)   |       |       | (GB)           | (s)        |
+================+========+=======+=======+================+============+
| GSplat-7k      | 27.61  | 0.84  | 0.18  | 4.52           | 159.96     |
+----------------+--------+-------+-------+----------------+------------+
| GSplat-30k     | 29.15  | 0.87  | 0.12  | 6.31           | 872.00     |
+----------------+--------+-------+-------+----------------+------------+

Performance metrics terms
--------------------------------------------------------------------

- **PSNR (Peak Signal-to-Noise Ratio):** Ratio between the maximum possible signal power and the power of corrupting noise. Higher values indicate better reconstruction quality. Typically, values above 30 dB represent good quality.  
- **SSIM (Structural Similarity Index):** Measures the similarity between two images considering luminance, contrast, and structure. Ranges from -1 to 1, where 1 indicates perfect similarity. SSIM is more perceptually aligned than PSNR.  
- **LPIPS (Learned Perceptual Image Patch Similarity):** Uses a neural network (AlexNet or VGG) to compute perceptual similarity between images. Lower values indicate more perceptually similar images.  It's considered to better align with human perception than PSNR or SSIM.
- **Number of Gaussians and rendering time:** Provides insights into model efficiency.

Scene-wise training memory and time
--------------------------------------------------------------------

The following tables show training time and memory for each scene.
The goal is to train faster with less GPU memory.

Train memory (GB):

+-------------+---------+--------+---------+--------+---------+--------+--------+
| Model       | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
+=============+=========+========+=========+========+=========+========+========+
| GSplat-7k   | 6.70    | 2.85   | 2.82    | 7.26   | 3.04    | 2.81   | 6.16   |
+-------------+---------+--------+---------+--------+---------+--------+--------+
| GSplat-30k  | 11.59   | 2.95   | 2.82    | 11.06  | 3.59    | 3.61   | 8.56   |
+-------------+---------+--------+---------+--------+---------+--------+--------+

Train time (s):

+-------------+---------+--------+---------+--------+---------+--------+--------+
| Model       | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
+=============+=========+========+=========+========+=========+========+========+
| GSplat-7k   | 147.44  | 161.15 | 169.35  | 168.50 | 175.19  | 155.99 | 142.11 |
+-------------+---------+--------+---------+--------+---------+--------+--------+
| GSplat-30k  | 1102.74 | 719.39 | 771.37  | 1067.28| 814.53  | 758.39 | 870.30 |
+-------------+---------+--------+---------+--------+---------+--------+--------+

Reproduced metrics per scene
--------------------------------------------------------------------

PSNR (dB):

+-------------+---------+--------+---------+--------+---------+--------+--------+
| Model       | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
+=============+=========+========+=========+========+=========+========+========+
| GSplat-7k   | 23.69   | 30.15  | 27.57   | 26.59  | 29.42   | 29.97  | 25.90  |
+-------------+---------+--------+---------+--------+---------+--------+--------+
| GSplat-30k  | 24.93   | 32.26  | 29.19   | 27.63  | 31.54   | 31.75  | 26.78  |
+-------------+---------+--------+---------+--------+---------+--------+--------+

SSIM:

+-------------+---------+--------+---------+--------+---------+--------+--------+
| Model       | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
+=============+=========+========+=========+========+=========+========+========+
| GSplat-7k   | 0.67    | 0.93   | 0.89    | 0.83   | 0.91    | 0.90   | 0.73   |
+-------------+---------+--------+---------+--------+---------+--------+--------+
| GSplat-30k  | 0.76    | 0.94   | 0.91    | 0.87   | 0.93    | 0.92   | 0.77   |
+-------------+---------+--------+---------+--------+---------+--------+--------+

LPIPS:

+-------------+---------+--------+---------+--------+---------+--------+--------+
| Model       | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
+=============+=========+========+=========+========+=========+========+========+
| GSplat-7k   | 0.30    | 0.13   | 0.18    | 0.11   | 0.11    | 0.19   | 0.23   |
+-------------+---------+--------+---------+--------+---------+--------+--------+
| GSplat-30k  | 0.16    | 0.11   | 0.13    | 0.07   | 0.08    | 0.14   | 0.14   |
+-------------+---------+--------+---------+--------+---------+--------+--------+

Number of Gaussians:

+-------------+---------+--------+---------+--------+---------+--------+--------+
| Model       | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
+=============+=========+========+=========+========+=========+========+========+
| GSplat-7k   | 4.46M   | 1.55M  | 1.64M   | 4.83M  | 1.89M   | 1.48M  | 4.11M  |
+-------------+---------+--------+---------+--------+---------+--------+--------+
| GSplat-30k  | 7.78M   | 1.85M  | 1.30M   | 6.69M  | 2.12M   | 2.29M  | 5.73M  |
+-------------+---------+--------+---------+--------+---------+--------+--------+

Summary
====================================================================

GSplat-7k trains faster with less GPU memory, while GSplat-30k achieves higher ``PSNR`` and ``SSIM`` metrics but requires more memory and time.  
This evaluation demonstrates the trade-offs between model size, training efficiency, and reconstruction quality on the Mip-NeRF 360 dataset.
