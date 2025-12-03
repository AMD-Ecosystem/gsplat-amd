.. meta::
  :description: GSplat reference
  :keywords: GSplat, ROCm, install, reference, API

.. _gsplat-api-reference:

********************************************************************
GSplat API reference
********************************************************************

This topic contains the GSplat API reference, listed by supported features.

Supported features
====================================================================

- :ref:`rasterization-1`

  - :ref:`rasterization-3dgs`

  - :ref:`rasterization-2dgs`

  - :ref:`densification-1`

    - :ref:`dens-default-strategy`

    - :ref:`mcmc-strategy`

  - :ref:`utils-1`

    - :ref:`utils-3dgs`

      - :ref:`3dgs-spherical-harmonics`

      - :ref:`quat-scale-to-covar-preci`

      - :ref:`utils-proj-3dgs`

      - :ref:`world-to-cam`

      - :ref:`fully-fused-projection-3dgs`

      - :ref:`isect-tiles-3dgs`

      - :ref:`isect-offset-encode-3dgs`

      - :ref:`rasterize-to-pixels-3dgs`

      - :ref:`rasterize-to-indices-in-range-3dgs`

      - :ref:`accumulate-3dgs`

      - :ref:`rasterization-3dgs-inria-wrapper`

    - :ref:`utils-2dgs`

      - :ref:`fully-fused-projection-2dgs`

      - :ref:`rasterize-to-pixels-2dgs`

      - :ref:`rasterize-to-indices-in-range-2dgs`

      - :ref:`accumulate-2dgs`

      - :ref:`rasterization-2dgs-inria-wrapper`

- :ref:`compression-support`


.. _rasterization-1:

Rasterization
====================================================================

.. _rasterization-3dgs:

3DGS
--------------------------------------------------------------------

.. code-block:: python

    rasterization(
        means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
        viewmats: Tensor, Ks: Tensor, width: int, height: int,
        near_plane: float = 0.01, far_plane: float = 1e10,
        radius_clip: float = 0.0, eps2d: float = 0.3,
        sh_degree: int | None = None, packed: bool = True, tile_size: int = 16,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB","D","ED","RGB+D","RGB+ED"] = "RGB",
        sparse_grad: bool = False, absgrad: bool = False,
        rasterize_mode: Literal["classic","antialiased"] = "classic",
        channel_chunk: int = 32, distributed: bool = False,
        camera_model: Literal["pinhole","ortho","fisheye","ftheta"] = "pinhole",
        segmented: bool = False, covars: Tensor | None = None,
        with_ut: bool = False, with_eval3d: bool = False,
        radial_coeffs: Tensor | None = None, tangential_coeffs: Tensor | None = None,
        thin_prism_coeffs: Tensor | None = None,
        ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
        rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
        viewmats_rs: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Dict]

- **Function**: Rasterizes **3D Gaussians (N)** into a batch of **image planes (C)**.

- **Returns**:

  - **render_colors**: The rendered colors. […, C, height, width, X]. X
    depends on the render_mode and input colors. If render_mode is “RGB”,
    X is D; if render_mode is “D” or “ED”, X is 1; if render_mode is
    “RGB+D” or “RGB+ED”, X is D+1.

  - **render_alphas**: The rendered alphas. […, C, height, width, 1].

  - **meta**: A dictionary of intermediate results of the rasterization.

- **Return type**: tuple

- **Parameters**:

  - **means** – The 3D centers of the Gaussians. […, N, 3]

  - **quats** – The quaternions of the Gaussians (wxyz convension). It’s
    not required to be normalized. […, N, 4]

  - **scales** – The scales of the Gaussians. […, N, 3]

  - **opacities** – The opacities of the Gaussians. […, N]

  - **colors** – The colors of the Gaussians. […, (C,) N, D] or […, (C,)
    N, K, 3] for SH coefficients.

  - **viewmats** – The world-to-cam transformation of the cameras. […,
    C, 4, 4]

  - **Ks** – The camera intrinsics. […, C, 3, 3]

  - **width** – The width of the image.

  - **height** – The height of the image.

  - **near_plane** – The near plane for clipping. Default is 0.01.

  - **far_plane** – The far plane for clipping. Default is 1e10.

  - **radius_clip** – Gaussians with 2D radius smaller or equal than
    this value will be skipped. This is extremely helpful for speeding
    up large scale scenes. Default is 0.0.

  - **eps2d** – An epsilon added to the egienvalues of projected 2D
    covariance matrices. This will prevents the projected GS to be too
    small. For example eps2d=0.3 leads to minimal 3 pixel unit. Default
    is 0.3.

  - **sh_degree** – The SH degree to use, which can be smaller than the
    total number of bands. If set, the colors should be […, (C,) N, K,
    3] SH coefficients, else the colors should be […, (C,) N, D]
    post-activation color values. Default is None.

  - **packed** – Whether to use packed mode which is more memory
    efficient but might or might not be as fast. Default is True.

  - **tile_size** – The size of the tiles for rasterization. Default is
    16. (Note: other values are not tested)

  - **backgrounds** – The background colors. […, C, D]. Default is None.

  - **render_mode** – The rendering mode. Supported modes are “RGB”,
    “D”, “ED”, “RGB+D”, and “RGB+ED”. “RGB” renders the colored image,
    “D” renders the accumulated depth, and “ED” renders the expected
    depth. Default is “RGB”.

  - **sparse_grad** – If true, the gradients for {means, quats, scales}
    will be stored in a COO sparse layout. This can be helpful for
    saving memory. Default is False.

  - **absgrad** – If true, the absolute gradients of the projected 2D
    means will be computed during the backward pass, which could be
    accessed by meta[“means2d”].absgrad. Default is False.

  - **rasterize_mode** – The rasterization mode. Supported modes are
    “classic” and “antialiased”. Default is “classic”.

  - **channel_chunk** – The number of channels to render in one go.
    Default is 32. If the required rendering channels are larger than
    this value, the rendering will be done looply in chunks.

  - **distributed** – Whether to use distributed rendering. Default is
    False. If True, The input Gaussians are expected to be a subset of
    scene in each rank, and the function will collaboratively render the
    images for all ranks.

  - **camera_model** – The camera model to use. Supported models are
    “pinhole”, “ortho”, “fisheye”, and “ftheta”. Default is “pinhole”.

  - **segmented** – Whether to use segmented radix sort. Default is
    False. Segmented radix sort performs sorting in segments, which is
    more efficient for the sorting operation itself. However, since it
    requires offset indices as input, additional global memory access is
    needed, which results in slower overall performance in most use
    cases.

  - **covars** – Optional covariance matrices of the Gaussians. If
    provided, the quats and scales will be ignored. […, N, 3, 3],
    Default is None.

  - **with_ut** – Whether to use Unscented Transform (UT) for
    projection. Default is False.

  - **with_eval3d** – Whether to calculate Gaussian response in 3D world
    space, instead of 2D image space. Default is False.

  - **radial_coeffs** – Opencv pinhole/fisheye radial distortion
    coefficients. Default is None. For pinhole camera, the shape should
    be […, C, 6]. For fisheye camera, the shape should be […, C, 4].

  - **tangential_coeffs** – Opencv pinhole tangential distortion
    coefficients. Default is None. The shape should be […, C, 2] if
    provided.

  - **thin_prism_coeffs** – Opencv pinhole thin prism distortion
    coefficients. Default is None. The shape should be […, C, 4] if
    provided.

  - **ftheta_coeffs** – F-Theta camera distortion coefficients shared
    for all cameras. Default is None. See
    FThetaCameraDistortionParameters for details.

  - **rolling_shutter** – The rolling shutter type. Default
    RollingShutterType.GLOBAL means global shutter.

  - **viewmats_rs** – The second viewmat when rolling shutter is used.
    Default is None.

.. _raster-3dgs-notes:

Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  This function is currently not differentiable with regard to the camera
    intrinsics Ks.

2.  **Multi-GPU Distributed Rasterization**: This function can be used
    in a multi-GPU distributed scenario by setting distributed to True.
    When distributed is True, a subset of total Gaussians could be
    passed into this function in each rank, and the function will
    collaboratively render a set of images using Gaussians from all
    ranks. Note to achieve balanced computation, it is recommended (not
    enforced) to have similar number of Gaussians in each rank. But it
    is enforced that the number of cameras to be rendered in each rank is
    the same. The function will return the rendered images corresponds
    to the input cameras in each rank, and allows for gradients to flow
    back to the Gaussians living in other ranks. For the details, please
    refer to the paper `On Scaling Up 3D Gaussian Splatting
    Training <https://arxiv.org/abs/2406.18533>`__.

3.  **Batch Rasterization**: This function allows for rasterizing a set
    of 3D Gaussians to a batch of images in one go, by simplly providing
    the batched viewmats and Ks.

4.  **Support N-D Features**: If sh_degree is None, the colors is
    expected to be with shape […, N, D] or […, C, N, D], in which D is
    the channel of the features to be rendered. The computation is slow
    when D > 32 at the moment. If sh_degree is set, the colors is
    expected to be the SH coefficients with shape […, N, K, 3] or […, C,
    N, K, 3], where K is the number of SH bases.

5.  **Depth Rendering**: This function supports colors or/and depths via
    render_mode. The supported modes are “RGB”, “D”, “ED”, “RGB+D”, and
    “RGB+ED”. “RGB” renders the colored image that respects the colors
    argument. “D” renders the accumulated z-depth “ED” renders the expected z-depth.
    “RGB+D” and “RGB+ED” render both the colored image and the depth,
    in which the depth is the last channel of the output.

6.  **Memory-Speed Trade-off**: The packed argument provides a trade-off
    between memory footprint and runtime. If packed is True, the
    intermediate results are packed into sparse tensors, which is more
    memory efficient but might be slightly slower. This is especially
    helpful when the scene is large and each camera sees only a small
    portion of the scene. If packed is False, the intermediate results
    are with shape […, C, N, …], which is faster but might consume more
    memory.

7.  **Sparse Gradients**: If sparse_grad is True, the gradients for
    {means, quats, scales} will be stored in a `COO sparse
    layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`__.
    This can be helpful for saving memory for training when the scene is
    large and each iteration only activates a small portion of the
    Gaussians. Usually a sparse optimizer is required to work with
    sparse gradients, such as
    `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`__.
    This argument is only effective when packed is True.

8.  **Speed-up for Large Scenes**: The radius_clip argument is extremely
    helpful for speeding up large scale scenes or scenes with large
    depth of fields. Gaussians with 2D radius smaller or equal than this
    value (in pixel unit) will be skipped during rasterization. This
    will skip all the far-away Gaussians that are too small to be seen
    in the image. But be warned that if there are close-up Gaussians
    that are also below this threshold, they will also get skipped
    (which is rarely happened in practice). This is by default disabled
    by setting radius_clip to 0.0.

9.  **Antialiased Rendering**: If rasterize_mode is “antialiased”, the
    function will apply a view-dependent compensation factor to Gaussian
    opacities, where is the projected 2D covariance matrix and is the
    eps2d. This will make the rendered image more antialiased, as
    proposed in the paper `Mip-Splatting: Alias-free 3D Gaussian
    Splatting <https://arxiv.org/pdf/2311.16493>`__.

10. **AbsGrad**: If absgrad is True, the absolute gradients of the
    projected 2D means will be computed during the backward pass, which
    could be accessed by meta[“means2d”].absgrad. This is an
    implementation of the paper `AbsGS: Recovering Fine Details for 3D
    Gaussian Splatting <https://arxiv.org/abs/2404.10484>`__, which is
    shown to be more effective for splitting Gaussians during training.

11. **Camera Distortion and Rolling Shutter**: The function supports
    rendering with opencv distortion formula for pinhole and fisheye
    cameras (radial_coeffs, tangential_coeffs, thin_prism_coeffs). It
    also supports rolling shutter rendering with the rolling_shutter
    argument. This is referenced from the paper `3DGUT: Enabling
    Distorted Cameras and Secondary Rays in Gaussian
    Splatting <https://arxiv.org/abs/2412.12507>`__.

.. _rasterization-2dgs:

2DGS
--------------------------------------------------------------------

.. code-block:: python

    rasterization_2dgs(
        means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
        viewmats: Tensor, Ks: Tensor, width: int, height: int,
        near_plane: float = 0.01, far_plane: float = 1e10,
        radius_clip: float = 0.0, eps2d: float = 0.3,
        sh_degree: int | None = None, packed: bool = False, tile_size: int = 16,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB","D","ED","RGB+D","RGB+ED"] = "RGB",
        sparse_grad: bool = False, absgrad: bool = False,
        distloss: bool = False, depth_mode: Literal["expected","median"] = "expected"
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]

- **Function**: Rasterizes a set of **2D Gaussians (N)** to a **batch of image planes (C)**.
  This function supports a handful of features, similar to the
  rasterization() function.

- **Returns**:

  - **render_colors**: The rendered colors. […, C, height, width, X]. X
    depends on the render_mode and input colors. If render_mode is “RGB”,
    X is D; if render_mode is “D” or “ED”, X is 1; if render_mode is
    “RGB+D” or “RGB+ED”, X is D+1.

  - **render_alphas**: The rendered alphas. […, C, height, width, 1].

  - **render_normals**: The rendered normals. […, C, height, width, 3].

  - **surf_normals**: surface normal from depth. […, C, height, width, 3]

  - **render_distort**: The rendered distortions. […, C, height, width,
    1]. L1 version, different from L2 version in 2DGS paper.

  - **render_median**: The rendered median depth. […, C, height, width,
    1].

  - **meta**: A dictionary of intermediate results of the rasterization.

- **Return type**: tuple

- **Parameters**:

  - **means** – The 3D centers of the Gaussians. […, N, 3]

  - **quats** – The quaternions of the Gaussians (wxyz convension). It’s
    not required to be normalized. […, N, 4]

  - **scales** – The scales of the Gaussians. […, N, 3]

  - **opacities** – The opacities of the Gaussians. […, N]

  - **colors** – The colors of the Gaussians. […, (C,) N, D] or […, (C,)
    N, K, 3] for SH coefficients.

  - **viewmats** – The world-to-cam transformation of the cameras. […,
    C, 4, 4]

  - **Ks** – The camera intrinsics. […, C, 3, 3]

  - **width** – The width of the image.

  - **height** – The height of the image.

  - **near_plane** – The near plane for clipping. Default is 0.01.

  - **far_plane** – The far plane for clipping. Default is 1e10.

  - **radius_clip** – Gaussians with 2D radius smaller or equal than
    this value will be skipped. This is extremely helpful for speeding
    up large scale scenes. Default is 0.0.

  - **eps2d** – An epsilon added to the egienvalues of projected 2D
    covariance matrices. This will prevents the projected GS to be too
    small. For example eps2d=0.3 leads to minimal 3 pixel unit. Default
    is 0.3.

  - **sh_degree** – The SH degree to use, which can be smaller than the
    total number of bands. If set, the colors should be [(C,) N, K, 3]
    SH coefficients, else the colors should [(C,) N, D] post-activation
    color values. Default is None.

  - **packed** – Whether to use packed mode which is more memory
    efficient but might or might not be as fast. Default is True.

  - **tile_size** – The size of the tiles for rasterization. Default is
    16. (Note: other values are not tested)

  - **backgrounds** – The background colors. [C, D]. Default is None.

  - **render_mode** – The rendering mode. Supported modes are “RGB”,
    “D”, “ED”, “RGB+D”, and “RGB+ED”. “RGB” renders the colored image,
    “D” renders the accumulated depth, and “ED” renders the expected
    depth. Default is “RGB”.

  - **sparse_grad** (*Experimental*) – If true, the gradients for
    {means, quats, scales} will be stored in a COO sparse layout. This
    can be helpful for saving memory. Default is False.

  - **absgrad** – If true, the absolute gradients of the projected 2D
    means will be computed during the backward pass, which could be
    accessed by meta[“means2d”].absgrad. Default is False.

  - **channel_chunk** – The number of channels to render in one go.
    Default is 32. If the required rendering channels are larger than
    this value, the rendering will be done looply in chunks.

  - **distloss** – If true, use distortion regularization to get better
    geometry detail.

  - **depth_mode** – render depth mode. Choose from expected depth and
    median depth.

.. _raster-2dgs-notes:

Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  This function is currently not differentiable with regard to the camera
    intrinsics Ks.

.. _densification-1:

Densification
===================================================================

In GSplat, you can abstract out the densification and pruning process of the
Gaussian training into a strategy. A strategy is a class that defines
how the Gaussian parameters (along with their optimizers) should be
updated (splitting, pruning, etc.) during the training.

.. _dens-default-strategy:

Default strategy
-------------------------------------------------------------------

.. code-block:: python

    class DefaultStrategy(
        prune_opa: float = 0.005, grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01, grow_scale2d: float = 0.05,
        prune_scale3d: float = 0.1, prune_scale2d: float = 0.15,
        refine_scale2d_stop_iter: int = 0, refine_start_iter: int = 500,
        refine_stop_iter: int = 15000, reset_every: int = 3000,
        refine_every: int = 100, pause_refine_after_reset: int = 0,
        absgrad: bool = False, revised_opacity: bool = False,
        verbose: bool = False,
        key_for_gradient: Literal["means2d","gradient_2dgs"] = "means2d"
    )

A default strategy that follows the original 3DGS paper. The strategy
will:

- Periodically duplicate GSs with high image plane gradients and small
  scales.

- Periodically split GSs with high image plane gradients and large
  scales.

- Periodically prune GSs with low opacity.

- Periodically reset GSs to a lower opacity.

- **Parameters**:

  - **prune_opa**
    (`float <https://docs.python.org/3/library/functions.html#float>`__)
    – GSs with opacity below this value will be pruned. Default is
    0.005.

  - **grow_grad2d**
    (`float <https://docs.python.org/3/library/functions.html#float>`__)
    – GSs with image plane gradient above this value will be
    split/duplicated. Default is 0.0002.

  - **grow_scale3d**
    (`float <https://docs.python.org/3/library/functions.html#float>`__)
    – GSs with 3D scale (normalized by scene_scale) below this value
    will be duplicated. Above will be split. Default is 0.01.

  - **grow_scale2d**
    (`float <https://docs.python.org/3/library/functions.html#float>`__)
    – GSs with 2D scale (normalized by image resolution) above this
    value will be split. Default is 0.05.

  - **prune_scale3d**
    (`float <https://docs.python.org/3/library/functions.html#float>`__)
    – GSs with 3d scale (normalized by scene_scale) above this value
    will be pruned. Default is 0.1.

  - **prune_scale2d**
    (`float <https://docs.python.org/3/library/functions.html#float>`__)
    – GSs with 2d scale (normalized by image resolution) above this
    value will be pruned. Default is 0.15.

  - **refine_scale2d_stop_iter**
    (`int <https://docs.python.org/3/library/functions.html#int>`__) –
    Stop refining GSs based on 2d scale after this iteration. Default is
    0. Set to a positive value to enable this feature.

  - **refine_start_iter**
    (`int <https://docs.python.org/3/library/functions.html#int>`__) –
    Start refining GSs after this iteration. Default is 500.

  - **refine_stop_iter**
    (`int <https://docs.python.org/3/library/functions.html#int>`__) –
    Stop refining GSs after this iteration. Default is 15_000.

  - **reset_every**
    (`int <https://docs.python.org/3/library/functions.html#int>`__) –
    Reset opacities every this steps. Default is 3000.

  - **refine_every**
    (`int <https://docs.python.org/3/library/functions.html#int>`__) –
    Refine GSs every this steps. Default is 100.

  - **pause_refine_after_reset**
    (`int <https://docs.python.org/3/library/functions.html#int>`__) –
    Pause refining GSs until this number of steps after reset, Default
    is 0 (no pause at all) and one might want to set this number to the
    number of images in training set.

  - **absgrad**
    (`bool <https://docs.python.org/3/library/functions.html#bool>`__) –
    Use absolute gradients for GS splitting. Default is False.

  - **revised_opacity**
    (`bool <https://docs.python.org/3/library/functions.html#bool>`__) –
    Whether to use revised opacity heuristic from arXiv:2404.06109
    (experimental). Default is False.

  - **verbose**
    (`bool <https://docs.python.org/3/library/functions.html#bool>`__) –
    Whether to print verbose information. Default is False.

  - **key_for_gradient**
    (`str <https://docs.python.org/3/library/stdtypes.html#str>`__) –
    Which variable uses for densification strategy. 3DGS uses “means2d”
    gradient and 2DGS uses a similar gradient which stores in variable
    “gradient_2dgs”.

.. _dens-strategy-notes:

Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. If absgrad=True, it will use the absolute gradients instead of
   average gradients for GS duplicating & splitting, following the AbsGS
   paper `AbsGS: Recovering Fine Details for 3D Gaussian
   Splatting <https://arxiv.org/abs/2404.10484>`__. Which typically
   leads to better results but requires to set the grow_grad2d to a
   higher value, e.g., 0.0008. Also, the rasterization() function should
   be called with absgrad=True as well so that the absolute gradients
   are computed.

.. _mcmc-strategy:

Markov Chain Monte Carlo (MCMC) strategy
------------------------------------------------------------------------

.. code-block:: python

    class MCMCStrategy(
        cap_max: int = 1_000_000, noise_lr: float = 5e5,
        refine_start_iter: int = 500, refine_stop_iter: int = 25000,
        refine_every: int = 100, min_opacity: float = 0.005,
        verbose: bool = False
    )

Strategy that follows the paper:

`3D Gaussian Splatting as Markov Chain Monte
Carlo <https://arxiv.org/abs/2404.09591>`__

This strategy will:

- Periodically teleport GSs with low opacity to a place that has high
  opacity.

- Periodically introduce new GSs sampled based on the opacity
  distribution.

- Periodically perturb the GSs locations.

**Parameters**:

- **cap_max**
  (`int <https://docs.python.org/3/library/functions.html#int>`__) –
  Maximum number of GSs. Default to 1_000_000.

- **noise_lr**
  (`float <https://docs.python.org/3/library/functions.html#float>`__) –
  MCMC samping noise learning rate. Default to 5e5.

- **refine_start_iter**
  (`int <https://docs.python.org/3/library/functions.html#int>`__) –
  Start refining GSs after this iteration. Default to 500.

- **refine_stop_iter**
  (`int <https://docs.python.org/3/library/functions.html#int>`__) –
  Stop refining GSs after this iteration. Default to 25_000.

- **refine_every**
  (`int <https://docs.python.org/3/library/functions.html#int>`__) –
  Refine GSs every this steps. Default to 100.

- **min_opacity**
  (`float <https://docs.python.org/3/library/functions.html#float>`__) –
  GSs with opacity below this value will be pruned. Default to 0.005.

- **verbose**
  (`bool <https://docs.python.org/3/library/functions.html#bool>`__) –
  Whether to print verbose information. Default to False.

.. _utils-1:

Utils
---------------------------------------------------------------------------

Below are the basic functions that supports the rasterization.

.. _utils-3dgs:

3DGS
--------------------------------------------------------------------------

.. _3dgs-spherical-harmonics:

spherical_harmonics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    spherical_harmonics(
        degrees_to_use: int, dirs: Tensor, coeffs: Tensor, masks: Tensor | None = None
    ) -> Tensor

- **Function**: Computes spherical harmonics.

- **Returns**:

  - Spherical harmonics. […, 3]


.. _quat-scale-to-covar-preci:

quat_scale_to_covar_preci
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    quat_scale_to_covar_preci(
        quats: Tensor, scales: Tensor,
        compute_covar: bool = True, compute_preci: bool = True, triu: bool = False
    ) -> Tuple[Tensor | None, Tensor | None]

- **Function**: Converts quaternions and scales to covariance and precision matrices.

- **Returns**:

  - **Covariance matrices**. If triu is True the returned shape is […,
    6], otherwise […, 3, 3].

  - **Precision matrices**. If triu is True the returned shape is […, 6],
    otherwise […, 3, 3].

- Return type: tuple

- **Parameters**:

  - **degrees_to_use** – The degree to be used.

  - **dirs** – Directions. […, 3]

  - **coeffs** – Coefficients. […, K, 3]

  - **masks** – Optional boolen masks to skip some computation. […,]
    Default: None.


.. _utils-proj-3dgs:

proj
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    proj(
        means: Tensor, covars: Tensor, Ks: Tensor,
        width: int, height: int,
        camera_model: Literal["pinhole","ortho","fisheye","ftheta"] = "pinhole"
    ) -> Tuple[Tensor, Tensor]

- **Function**: Projection of Gaussians (perspective or orthographic).

- **Returns**:

  - **Projected means**. […, C, N, 2]

  - **Projected covariances**. […, C, N, 2, 2]

- **Return type**: A tuple

- **Parameters**:

  - **means** – Gaussian means. […, C, N, 3]

  - **covars** – Gaussian covariances. […, C, N, 3, 3]

  - **Ks** – Camera intrinsics. […, C, 3, 3]

  - **width** – Image width.

  - **height** – Image height.


.. _world-to-cam:

world_to_cam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    world_to_cam(means: Tensor, covars: Tensor, viewmats: Tensor) -> Tuple[Tensor, Tensor]


- **Function**: Transforms Gaussians from world to camera coordinate system.

- **Returns**:

  - **Gaussian means in camera coordinate system**. […, C, N, 3]

  - **Gaussian covariances in camera coordinate system**. […, C, N, 3, 3]

- **Return type**: tuple

- **Parameters**:

  - **means** – Gaussian means. […, N, 3]

  - **covars** – Gaussian covariances. […, N, 3, 3]

  - **viewmats** – World-to-camera transformation matrices. […, C, 4, 4]


.. _fully-fused-projection-3dgs:

fully_fused_projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    fully_fused_projection(
        means: Tensor, covars: Tensor | None, quats: Tensor | None, scales: Tensor | None,
        viewmats: Tensor, Ks: Tensor, width: int, height: int,
        eps2d: float = 0.3, near_plane: float = 0.01, far_plane: float = 1e10,
        radius_clip: float = 0.0, packed: bool = False, sparse_grad: bool = False,
        calc_compensations: bool = False,
        camera_model: Literal["pinhole","ortho","fisheye","ftheta"] = "pinhole",
        opacities: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

- **Function**: Projects Gaussians to 2D. This function fuse the process of computing
  covariances (quat_scale_to_covar_preci()), transforming to camera
  space (world_to_cam()), and projection (proj()).

- **Returns**:

  - If packed is True:

    - **batch_ids**. The batch indices of the projected Gaussians. Int32
      tensor of shape [nnz].

    - **camera_ids**. The camera indices of the projected Gaussians. Int32
      tensor of shape [nnz].

    - **gaussian_ids**. The column indices of the projected Gaussians.
      Int32 tensor of shape [nnz].

    - **radii**. The maximum radius of the projected Gaussians in pixel
      unit. Int32 tensor of shape [nnz, 2].

    - **means**. Projected Gaussian means in 2D. [nnz, 2]

    - **depths**. The z-depth of the projected Gaussians. [nnz]

    - **conics**. Inverse of the projected covariances. Return the flattend
      upper triangle with [nnz, 3]

    - **compensations**. The view-dependent opacity compensation factor.
      [nnz]

  - If packed is False:

    - **radii**. The maximum radius of the projected Gaussians in pixel
      unit. Int32 tensor of shape […, C, N, 2].

    - **means**. Projected Gaussian means in 2D. […, C, N, 2]

    - **depths**. The z-depth of the projected Gaussians. […, C, N]

    - **conics**. Inverse of the projected covariances. Return the flattend
      upper triangle with […, C, N, 3]

    - **compensations**. The view-dependent opacity compensation factor.
      […, C, N]

- **Return type**: tuple

- **Parameters**:

  - **means2d** – Projected Gaussian means. […, N, 2] if packed is
    False, [nnz, 2] if packed is True.

  - **conics** – Inverse of the projected covariances with only upper
    triangle values. […, N, 3] if packed is False, [nnz, 3] if packed is
    True.

  - **colors** – Gaussian colors or ND features. […, N, channels] if
    packed is False, [nnz, channels] if packed is True.

  - **opacities** – Gaussian opacities that support per-view values. […,
    N] if packed is False, [nnz] if packed is True.

  - **image_width** – Image width.

  - **image_height** – Image height.

  - **tile_size** – Tile size.

  - **isect_offsets** – Intersection offsets outputs from
    isect_offset_encode(). […, tile_height, tile_width]

  - **flatten_ids** – The global flatten indices in [I \* N] or [nnz]
    from isect_tiles(). [n_isects]

  - **backgrounds** – Background colors. […, channels]. Default: None.

  - **masks** – Optional tile mask to skip rendering GS to masked tiles.
    […, tile_height, tile_width]. Default: None.

  - **packed** – If True, the input tensors are expected to be packed
    with shape [nnz, …]. Default: False.

  - **absgrad** – If True, the backward pass will compute a .absgrad
    attribute for means2d. Default: False.

.. _fully-fused-3dgs-notes:

Notes:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. During projection, the Gaussians that are outside of the camera
   frustum are ignored. So not all the elements in the output tensors are
   valid. The output radii could serve as an indicator, in which zero
   radii means the corresponding elements are invalid in the output
   tensors and will be ignored in the next rasterization process. If
   packed=True, the output tensors will be packed into a flattened
   tensor, in which all elements are valid. In this case, a
   ``batch_ids`` tensor and ``camera_ids`` tensor will be returned to indicate the batch,
   camera and gaussian indices of the packed flattened tensor, which is
   essentially following the COO sparse tensor format.

2. This functions supports projecting Gaussians with either covariances
   or {quaternions, scales}, which will be converted to covariances
   internally in a fused CUDA kernel. Either covars or {quats, scales}
   should be provided.

.. _isect-tiles-3dgs:

isect_tiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    isect_tiles(
        means2d: Tensor, radii: Tensor, depths: Tensor,
        tile_size: int, tile_width: int, tile_height: int,
        sort: bool = True, segmented: bool = False, packed: bool = False,
        n_images: int | None = None, image_ids: Tensor | None = None, gaussian_ids: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]

- **Function**:Maps projected Gaussians to intersecting tiles.

- **Returns**:

  - **Tiles per Gaussian**. The number of tiles intersected by each
    Gaussian. Int32 […, N] if packed is False, Int32 [nnz] if packed is
    True.

  - **Intersection ids**. Each id is an 64-bit integer with the following
    information: image_id (Xc bits) \| tile_id (Xt bits) \| depth (32 bits).
    Xc and Xt are the maximum number of bits required to represent the image
    and tile ids, respectively. Int64 [n_isects]

  - **Flatten ids**. The global flatten indices in [I \* N] or [nnz]
    (packed). [n_isects]

- **Return type**: tuple

- **Parameters**:

  - **means2d** – Projected Gaussian means. […, N, 2] if packed is
    False, [nnz, 2] if packed is True.

  - **radii** – Maximum radii of the projected Gaussians. […, N, 2] if
    packed is False, [nnz, 2] if packed is True.

  - **depths** – Z-depth of the projected Gaussians. […, N] if packed is
    False, [nnz] if packed is True.

  - **tile_size** – Tile size.

  - **tile_width** – Tile width.

  - **tile_height** – Tile height.

  - **sort** – If True, the returned intersections will be sorted by the
    intersection ids. Default: True.

  - **segmented** – If True, segmented radix sort will be used to sort
    the intersections. Default: False.

  - **packed** – If True, the input tensors are packed. Default: False.

  - **n_images** – Number of images. Required if packed is True.

  - **image_ids** – The image indices of the projected Gaussians.
    Required if packed is True.

  - **gaussian_ids** – The column indices of the projected Gaussians.
    Required if packed is True.


.. _isect-offset-encode-3dgs:

isect_offset_encode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    isect_offset_encode(isect_ids: Tensor, n_images: int, tile_width: int, tile_height: int) -> Tensor

- **Function**: Encodes intersection ids to offsets.

- **Returns**: Offsets. [I, tile_height, tile_width]

- **Parameters**:

  - **isect_ids** – Intersection ids. [n_isects]

  - **n_images** – Number of images.

  - **tile_width** – Tile width.

  - **tile_height** – Tile height.


.. _rasterize-to-pixels-3dgs:

rasterize_to_pixels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    rasterize_to_pixels(
        means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor,
        image_width: int, image_height: int, tile_size: int,
        isect_offsets: Tensor, flatten_ids: Tensor,
        backgrounds: Tensor | None = None, masks: Tensor | None = None,
        packed: bool = False, absgrad: bool = False
    ) -> Tuple[Tensor, Tensor]

- **Function**: Rasterizes Gaussians to pixels.

- **Returns**:

  - **Rendered colors**. […, image_height, image_width, channels]

  - **Rendered alphas**. […, image_height, image_width, 1]

- **Return type**: tuple

- **Parameters**:

  - **means2d** – Projected Gaussian means. […, N, 2] if packed is
    False, [nnz, 2] if packed is True.

  - **conics** – Inverse of the projected covariances with only upper
    triangle values. […, N, 3] if packed is False, [nnz, 3] if packed is
    True.

  - **colors** – Gaussian colors or ND features. […, N, channels] if
    packed is False, [nnz, channels] if packed is True.

  - **opacities** – Gaussian opacities that support per-view values. […,
    N] if packed is False, [nnz] if packed is True.

  - **image_width** – Image width.

  - **image_height** – Image height.

  - **tile_size** – Tile size.

  - **isect_offsets** – Intersection offsets outputs from
    isect_offset_encode(). […, tile_height, tile_width]

  - **flatten_ids** – The global flatten indices in [I \* N] or [nnz]
    from isect_tiles(). [n_isects]

  - **backgrounds** – Background colors. […, channels]. Default: None.

  - **masks** – Optional tile mask to skip rendering GS to masked tiles.
    […, tile_height, tile_width]. Default: None.

  - **packed** – If True, the input tensors are expected to be packed
    with shape [nnz, …]. Default: False.

  - **absgrad** – If True, the backward pass will compute a .absgrad
    attribute for means2d. Default: False.


.. _rasterize-to-indices-in-range-3dgs:

rasterize_to_indices_in_range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    rasterize_to_indices_in_range(
        range_start: int, range_end: int, transmittances: Tensor,
        means2d: Tensor, conics: Tensor, opacities: Tensor,
        image_width: int, image_height: int, tile_size: int,
        isect_offsets: Tensor, flatten_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]

- **Function**: Rasterizes a batch of Gaussians to images but only returns the
  indices.

- **Returns**:

  - **Gaussian ids**. Gaussian ids for the pixel intersection. A
    flattened list of shape [M].

  - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].

  - **Image ids**. image indices. A flattened list of shape [M].

- **Return type**: tuple

- **Parameters**:

  - **range_start** – The start batch of Gaussians to be rasterized
    (inclusive).

  - **range_end** – The end batch of Gaussians to be rasterized
    (exclusive).

  - **transmittances** – Currently transmittances. […, image_height,
    image_width]

  - **means2d** – Projected Gaussian means. […, N, 2]

  - **conics** – Inverse of the projected covariances with only upper
    triangle values. […, N, 3]

  - **opacities** – Gaussian opacities that support per-view values. […,
    N]

  - **image_width** – Image width.

  - **image_height** – Image height.

  - **tile_size** – Tile size.

  - **isect_offsets** – Intersection offsets outputs from
    isect_offset_encode(). […, tile_height, tile_width]

  - **flatten_ids** – The global flatten indices in [I \* N] from
    isect_tiles(). [n_isects]


.. _rasterize-indices-notes:

Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. This function supports iterative rasterization, in which each call of
   this function will rasterize a batch of Gaussians from near to far,
   defined by [range_start, range_end). If a one-step full rasterization
   is desired, set range_start to 0 and range_end to a really large
   number, e.g, 1e10.


.. _accumulate-3dgs:

accumulate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    accumulate(
        means2d: Tensor, conics: Tensor, opacities: Tensor, colors: Tensor,
        gaussian_ids: Tensor, pixel_ids: Tensor, image_ids: Tensor,
        image_width: int, image_height: int
    ) -> Tuple[Tensor, Tensor]

- **Function**: Alpah compositing of 2D Gaussians in Pure Pytorch. This function
  performs alpha compositing for Gaussians based on the pair of indices
  {gaussian_ids, pixel_ids, image_ids}, which annotates the intersection
  between all pixels and Gaussians. These intersections can be accquired
  from gsplat.rasterize_to_indices_in_range.

- **Returns**:

  - **renders**: Accumulated colors. […, image_height, image_width, 
    channels]

  - **alphas**: Accumulated opacities. […, image_height, image_width, 1]

- **Return type**: tuple

- **Parameters**:

  - **means2d** – Gaussian means in 2D. […, N, 2]

  - **conics** – Inverse of the 2D Gaussian covariance, Only upper
    triangle values. […, N, 3]

  - **opacities** – Per-view Gaussian opacities (for example, when
    antialiasing is enabled, Gaussian in each view would efficiently
    have different opacity). […, N]

  - **colors** – Per-view Gaussian colors. Supports N-D features. […, N,
    channels]

  - **gaussian_ids** – Collection of Gaussian indices to be rasterized.
    A flattened list of shape [M].

  - **pixel_ids** – Collection of pixel indices (row-major) to be
    rasterized. A flattened list of shape [M].

  - **image_ids** – Collection of image indices to be rasterized. A
    flattened list of shape [M].

  - **image_width** – Image width.

  - **image_height** – Image height.


.. _accumulate-3dgs-notes:

Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. This function exposes the alpha compositing process into pure
   Pytorch. So it relies on Pytorch’s autograd for the backpropagation.
   It is much slower than our fully fused rasterization implementation
   and comsumes much more GPU memory. But it could serve as a playground
   for new ideas or debugging, as no backward implementation is needed.

2. This function requires the nerfacc package to be installed.


.. _rasterization-3dgs-inria-wrapper:

rasterization_inria_wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    rasterization_inria_wrapper(
        means: Tensor, quats: Tensor, scales:
        Tensor, opacities: Tensor, colors: Tensor, viewmats: Tensor, Ks: Tensor,
        width: int, height: int, near_plane: float = 0.01, far_plane: float =
        100.0, eps2d: float = 0.3, sh_degree: int \| None = None, backgrounds:
        Tensor \| None = None, \**kwargs
    ) → Tuple[Tensor, Tensor, Dict][source]

- **Function**: Wrapper for Inria’s rasterization backend. This function exists for
  comparison purpose only. Only rendered image is returned.

.. _utils-2dgs:

2DGS
------------------------------------------------------------------------


.. _fully-fused-projection-2dgs:

fully_fused_projection_2dgs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    fully_fused_projection_2dgs(
        means: Tensor, quats: Tensor, scales: Tensor,
        viewmats: Tensor, Ks: Tensor, width: int, height: int,
        eps2d: float = 0.3, near_plane: float = 0.01, far_plane: float = 1e10,
        radius_clip: float = 0.0, packed: bool = False, sparse_grad: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]

- **Function**: Prepare Gaussians for rasterization. This function prepares ray-splat
  intersection matrices, computes per splat bounding box and 2D means in
  image space.

- **Returns**:

  - If packed is True:

    - **batch_ids**. The batch indices of the projected Gaussians. Int32
      tensor of shape [nnz].

    - **camera_ids**. The camera indices of the projected Gaussians. Int32
      tensor of shape [nnz].

    - **gaussian_ids**. The column indices of the projected Gaussians.
      Int32 tensor of shape [nnz].

    - **radii**. The maximum radius of the projected Gaussians in pixel
      unit. Int32 tensor of shape [nnz, 2].

    - **means**. Projected Gaussian means in 2D. [nnz, 2]

    - **depths**. The z-depth of the projected Gaussians. [nnz]

    - **ray_transforms**. transformation matrices that transforms xy-planes
      in pixel spaces into splat coordinates (WH)^T in equation (9) in
      paper [nnz, 3, 3]

    - **normals**. The normals in camera spaces. [nnz, 3]

  - If packed is False:

    - **radii**. The maximum radius of the projected Gaussians in pixel
      unit. Int32 tensor of shape […, C, N, 2].

    - **means**. Projected Gaussian means in 2D. […, C, N, 2]

    - **depths**. The z-depth of the projected Gaussians. […, C, N]

    - **ray_transforms**. transformation matrices that transforms xy-planes
      in pixel spaces into splat coordinates […, C, N, 3, 3]

    - **normals**. The normals in camera spaces. […, C, N, 3]

- **Return type**: tuple

- **Parameters**:

  - **means** – Gaussian means. […, N, 3]

  - **quats** – Quaternions (No need to be normalized). […, N, 4].

  - **scales** – Scales. […, N, 3].

  - **viewmats** – World-to-camera matrices. […, C, 4, 4]

  - **Ks** – Camera intrinsics. […, C, 3, 3]

  - **width** – Image width.

  - **height** – Image height.

  - **near_plane** – Near plane distance. Default: 0.01.

  - **far_plane** – Far plane distance. Default: 200.

  - **radius_clip** – Gaussians with projected radii smaller than this
    value will be ignored. Default: 0.0.

  - **packed** – If True, the output tensors will be packed into a
    flattened tensor. Default: False.

  - **sparse_grad** (*Experimental*) – This is only effective when
    packed is True. If True, during backward the gradients of {means,
    covars, quats, scales} will be a sparse Tensor in COO layout.
    Default: False.


.. _rasterize-to-pixels-2dgs:

rasterize_to_pixels_2dgs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    rasterize_to_pixels_2dgs(
        means2d: Tensor, ray_transforms: Tensor, colors: Tensor, opacities: Tensor,
        normals: Tensor, densify: Tensor, image_width: int, image_height: int,
        tile_size: int, isect_offsets: Tensor, flatten_ids: Tensor,
        backgrounds: Tensor | None = None, masks: Tensor | None = None,
        packed: bool = False, absgrad: bool = False, distloss: bool = False
    ) -> Tuple[Tensor, Tensor]

- **Function**: Rasterize Gaussians to pixels.

- **Returns**:

  - **Rendered colors**. […, image_height, image_width, channels]

  - **Rendered alphas**. […, image_height, image_width, 1]

  - **Rendered normals**. […, image_height, image_width, 3]

  - **Rendered distortion**. […, image_height, image_width, 1]

  - **Rendered median depth**.[…, image_height, image_width, 1]

- **Return type**: tuple

- **Parameters**:

  - **means2d** – Projected Gaussian means. […, N, 2] if packed is
    False, [nnz, 2] if packed is True.

  - **ray_transforms** – transformation matrices that transforms
    xy-planes in pixel spaces into splat coordinates. […, N, 3, 3] if
    packed is False, [nnz, channels] if packed is True.

  - **colors** – Gaussian colors or ND features. […, N, channels] if
    packed is False, [nnz, channels] if packed is True.

  - **opacities** – Gaussian opacities that support per-view values. […,
    N] if packed is False, [nnz] if packed is True.

  - **normals** – The normals in camera space. […, N, 3] if packed is
    False, [nnz, 3] if packed is True.

  - **densify** – Dummy variable to keep track of gradient for
    densification. […, N, 2] if packed, [nnz, 3] if packed is True.

  - **tile_size** – Tile size.

  - **isect_offsets** – Intersection offsets outputs from
    isect_offset_encode(). […, tile_height, tile_width]

  - **flatten_ids** – The global flatten indices in [I \* N] or [nnz]
    from isect_tiles(). [n_isects]

  - **backgrounds** – Background colors. […, channels]. Default: None.

  - **masks** – Optional tile mask to skip rendering GS to masked tiles.
    […, tile_height, tile_width]. Default: None.

  - **packed** – If True, the input tensors are expected to be packed
    with shape [nnz, …]. Default: False.

  - **absgrad** – If True, the backward pass will compute a .absgrad
    attribute for means2d. Default: False.


.. _rasterize-to-indices-in-range-2dgs:

rasterize_to_indices_in_range_2dgs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    rasterize_to_indices_in_range_2dgs(
        range_start: int, range_end: int, transmittances: Tensor,
        means2d: Tensor, ray_transforms: Tensor, opacities: Tensor,
        image_width: int, image_height: int, tile_size: int,
        isect_offsets: Tensor, flatten_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]

- **Function**: Rasterizes a batch of Gaussians to images but only returns the
  indices. This function supports iterative rasterization, in which each
  call of this function will rasterize a batch of Gaussians from near to
  far, defined by [range_start, range_end). If a one-step full
  rasterization is desired, set range_start to 0 and range_end to a
  really large number, e.g, 1e10.

- **Returns**:

  - **Gaussian ids**. Gaussian ids for the pixel intersection. A
    flattened list of shape [M].

  - **Pixel ids**. pixel indices (row-major). A flattened list of shape
    [M].

  - **Camera ids**. Camera indices. A flattened list of shape [M].

  - **Batch ids**. Batch indices. A flattened list of shape [M].

- **Return type**: tuple

- **Parameters**:

  - **range_start** – The start batch of Gaussians to be rasterized
    (inclusive).

  - **range_end** – The end batch of Gaussians to be rasterized
    (exclusive).

  - **transmittances** – Currently transmittances. […, image_height,
    image_width]

  - **means2d** – Projected Gaussian means. […, N, 2]

  - **ray_transforms** – transformation matrices that transforms
    xy-planes in pixel spaces into splat coordinates. […, N, 3, 3]

  - **opacities** – Gaussian opacities that support per-view values. […,
    N]

  - **image_width** – Image width.

  - **image_height** – Image height.

  - **tile_size** – Tile size.

  - **isect_offsets** – Intersection offsets outputs from
    isect_offset_encode(). […, tile_height, tile_width]

  - **flatten_ids** – The global flatten indices in [I \* N] from
    isect_tiles(). [n_isects]


.. _accumulate-2dgs:

accumulate_2dgs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    accumulate_2dgs(
        means2d: Tensor, ray_transforms: Tensor, opacities:
        Tensor, colors: Tensor, normals: Tensor, gaussian_ids: Tensor,
        pixel_ids: Tensor, image_ids: Tensor, image_width: int, image_height: int
    ) → Tuple[Tensor, Tensor, Tensor][source]

- **Function**: Alpha compositing for 2DGS.
- **Returns**:

  - **renders**: Accumulated colors. […, image_height, image_width,
    channels]

  - **alphas**: Accumulated opacities. […, image_height, image_width, 1]

  - **normals**: Accumulated normals. […, image_height, image_width, 3]

- **Return type**: tuple

- **Parameters**:

  - **means2d** – Gaussian means in 2D. [C, N, 2]

  - **ray_transforms** – transformation matrices that transform rays in
    pixel space into splat’s local frame. [C, N, 3, 3]

  - **opacities** – Per-view Gaussian opacities (for example, when
    antialiasing is enabled, Gaussian in each view would efficiently
    have different opacity). [C, N]

  - **colors** – Per-view Gaussian colors. Supports N-D features. [C, N,
    channels]

  - **normals** – Per-view Gaussian normals. [C, N, 3]

  - **gaussian_ids** – Collection of Gaussian indices to be rasterized.
    A flattened list of shape [M].

  - **pixel_ids** – Collection of pixel indices (row-major) to be
    rasterized. A flattened list of shape [M].

  - **image_ids** – Collection of image indices to be rasterized. A
    flattened list of shape [M].

  - **image_width** – Image width.

  - **image_height** – Image height.


.. _accumulate-2dgs-notes:

Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. This function requires the `nerfacc <https://github.com/nerfstudio-project/nerfacc>`__ package to be installed.


.. _rasterization-2dgs-inria-wrapper:

rasterization_2dgs_inria_wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    rasterization_2dgs_inria_wrapper(
        means: Tensor, quats: Tensor, scales:
        Tensor, opacities: Tensor, colors: Tensor, viewmats: Tensor, Ks: Tensor,
        width: int, height: int, near_plane: float = 0.01, far_plane: float =
        100.0, eps2d: float = 0.3, sh_degree: int \| None = None, backgrounds:
        Tensor \| None = None, depth_ratio: int = 0, \**kwargs
    ) → Tuple[Tuple, Dict]


- **Function**: Wrapper for 2DGS’s rasterization backend which is based on Inria’s
  backend.


.. _compression-support:

Compression
=====================================================================

Compression of gaussian parameters is currently not supported.