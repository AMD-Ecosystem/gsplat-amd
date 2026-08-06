import glob
import os
import os.path as osp
import pathlib
import platform
import sys
import re

from setuptools import find_packages, setup

# Please run the rock_setup.sh script to set the environment variables for the ROCm SDK
IS_ROCM = True
ROCM_HOME = os.getenv("ROCM_PATH", "/opt/rocm")
import torch

__version__ = None
exec(open("gsplat/version.py", "r").read())
import subprocess

def get_rocm_arch():
    """
    Runs rocminfo and extracts the GPU architecture (gfx code).
    Returns:
        str: The gfx code (e.g., 'gfx942', 'gfx90a'), or 'gfx942' as fallback.
    """
    # Explicit override. rocminfo needs a working /dev/kfd, so on a build host
    # with no (or a temporarily unavailable) GPU the detection below silently
    # falls back to gfx942 -- which builds wave64 kernels that will never run on
    # a wave32 part. Set GSPLAT_GPU_ARCH=gfx1250 to pin it.
    env_arch = os.environ.get("GSPLAT_GPU_ARCH", "").strip()
    if env_arch:
        print(f"GSPLAT_GPU_ARCH override: {env_arch}")
        return env_arch
    try:
        # Run rocminfo command
        result = subprocess.run(
            ['rocminfo'],
            capture_output=True,
            text=True,
            check=True
        ) 
        # Parse the output to find the gfx architecture
        # Look for lines like "Name:                    gfx942"
        output = result.stdout
        # Search for gfx code pattern
        match = re.search(r'Name:\s+(gfx[0-9a-z]+)', output)
        if match:
            gfx_code = match.group(1)
            print(f"Detected ROCm GPU architecture: {gfx_code}")
            return gfx_code
        # Alternative pattern: sometimes it appears as "gfxXXX" directly
        match = re.search(r'\b(gfx[0-9a-z]+)\b', output)
        if match:
            gfx_code = match.group(1)
            print(f"Detected ROCm GPU architecture: {gfx_code}")
            return gfx_code
        print("Warning: Could not detect GPU architecture from rocminfo, using default gfx942")
        return "gfx942"
    except subprocess.CalledProcessError as e:
        print(f"Error running rocminfo: {e}")
        print("Using default architecture: gfx942")
        return "gfx942"
    except FileNotFoundError:
        print("Error: rocminfo not found. Make sure ROCm is installed and in PATH.")
        print("Using default architecture: gfx942")
        return "gfx942"
    except Exception as e:
        print(f"Unexpected error getting ROCm architecture: {e}")
        print("Using default architecture: gfx942")
        return "gfx942"

def is_git_repo(folder_path):
    """
    Checks if a folder is a Git repository by running 'git rev-parse --git-dir'.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        bool: True if it is a Git repository, False otherwise.
    """
    # First, check if the folder path is a valid directory
    if not os.path.isdir(folder_path):
        return False

    try:
        # Run the git command
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=folder_path,
            capture_output=True,
            text=True
        )

        # The command returns an exit code of 0 if it's a repo
        return result.returncode == 0

    except FileNotFoundError:
        # This exception is raised if 'git' is not in the system's PATH
        print("Error: Git is not installed or not in your system's PATH.")
        return False
    except Exception as e:
        # Handle other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return False

def get_git_rev(folder_path):
    """
    Checks if a folder is a Git repository and returns the latest commit SHA
    of the main branch using Git command-line tools.

    Args:
        folder_path (str): The path to the folder to check.

    Returns:
        str: The SHA of the latest commit on the main branch, or None if
             it's not a Git repository or the main branch doesn't exist.
    """
    try:
        # Use subprocess to run 'git rev-parse main' to get the commit hash
        # 'rev-parse' is a low-level command used to translate a human-readable
        # name into an SHA-1.
        command = ['git', 'rev-parse', '--short' ,"HEAD"]

        # Run the command in the specified folder
        result = subprocess.run(
            command,
            cwd=folder_path,
            capture_output=True,
            text=True,
            check=True
        )

        # The output is the commit hash
        commit_sha = result.stdout.strip()
        return commit_sha

    except subprocess.CalledProcessError as e:
        # This error is raised if the command fails, which happens if 'main'
        # branch doesn't exist
        print(f"Error: The #{branch_name} branch does not exist or another error occurred: {e}")
        return ""
    except FileNotFoundError:
        print("Error: Git is not installed or not in your system's PATH.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""
    return ""

if is_git_repo:
    git_rev = get_git_rev(os.getcwd())
    __version__ += f"+{git_rev}"

print(f"VERSION = {__version__}")

URL = "https://github.com/rocm/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS =  os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

ENABLE_TEST_COVERAGE = os.getenv("ENABLE_TEST_COVERAGE", "0") == "1"

MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")

def get_ext():
    from torch.utils.cpp_extension import BuildExtension
    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    if IS_ROCM:
        from torch.utils.cpp_extension import CUDAExtension
        print("ROCM detected, compiling with HIP support...")
        from torch.utils.cpp_extension import CppExtension
        # Get the GPU architecture dynamically
        gpu_arch = get_rocm_arch()
        print(f"gpu arch is set to {gpu_arch}")

        # Use relative path instead of hardcoded absolute path
        extensions_dir = osp.join("gsplat","cuda")
        sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(osp.join(extensions_dir, "csrc", "*.cpp"))
        sources += [osp.join(extensions_dir, "ext.cpp")]
        # Wavefront width: CDNA (gfx9xx, e.g. MI2xx/MI3xx) is wave64; RDNA and
        # CDNA5 (gfx10xx/11xx/12xx, e.g. gfx1250 / MI400) are wave32. gsplat's
        # reduction kernels are templated on this via GSPLAT_WARP_SIZE; pass it
        # to BOTH host and device compilation so launcher shmem sizing and the
        # in-kernel reductions agree. (GSPLAT_USE_WAVE64 is derived from it and
        # gates the wave64-only single-wave "bs64" DPP rasterizer path.)
        _arch_num = "".join(ch for ch in gpu_arch[3:] if ch.isdigit())
        gsplat_warp_size = 32 if _arch_num[:2] in ("10", "11", "12") else 64
        print(f"gsplat warp size set to {gsplat_warp_size} (for {gpu_arch})")

        undef_macros = []
        define_macros = []

        extra_compile_args = {"cxx": ["-D__HIP_PLATFORM_AMD__" , "-Wno-sign-compare", "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE", "-DUSE_ROCM"]}
        if WITH_SYMBOLS:
            extra_compile_args["cxx"] += ["-g", "-O0"]
        else:
            extra_compile_args = {"cxx": ["-O3", "-Wno-attributes", "-Wno-switch", "-Wno-comment"]}

        # Normally strip symbols (-s). When inspecting kernel resource usage,
        # keep symbols so the code object can be read back after the build.
        extra_link_args = [] if os.getenv("KERNEL_RESOURCE_USAGE", "0") == "1" else ["-s"]

        # Compile with OpenMP
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        extra_compile_args["cxx"] += ["-fopenmp"]

        # Keep host (launcher) compilation in sync with the device warp width.
        extra_compile_args["cxx"] += [f"-DGSPLAT_WARP_SIZE={gsplat_warp_size}"]

        hipcc_flags = [ "-D__HIP_PLATFORM_AMD__", "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE", "-DUSE_ROCM" , f"--offload-arch={gpu_arch}", f"-DGSPLAT_WARP_SIZE={gsplat_warp_size}"]
        # Emit hardware floating-point global/LDS atomics (global_atomic_add_f32)
        # instead of slow compare-and-swap (CAS) loops. The 3DGS backward is
        # dominated by gradient atomicAdds; on wave32 (gfx1250) the atomic count
        # is already doubled vs wave64, so fast HW atomics matter even more.
        hipcc_flags += ["-munsafe-fp-atomics"]
        # Match the CUDA path, which has always built with nvcc --use_fast_math
        # (see the nvcc_flags below). The HIP port never carried that across, so
        # a plain `1.0f / x` compiles to the ~12-instruction IEEE division
        # sequence (v_div_scale/v_div_fmas/v_div_fixup) instead of a single
        # v_rcp_f32. In the 3DGS backward that sequence sits on the critical
        # path of every gaussian, twice per lane. Set FAST_MATH=0 to A/B it.
        if os.getenv("FAST_MATH", "1") != "0":
            hipcc_flags += ["-ffast-math"]
        # Opt-in: print per-kernel VGPR/SGPR/spill/LDS/occupancy at compile time.
        # Build with KERNEL_RESOURCE_USAGE=1 and read the remarks in the build log.
        if os.getenv("KERNEL_RESOURCE_USAGE", "0") == "1":
            hipcc_flags += ["-Rpass-analysis=kernel-resource-usage"]
        if WITH_SYMBOLS:
            hipcc_flags += ["-g", "-ggdb" , "-O0"]
        else:
            hipcc_flags += ["-O3" ]
        # Opt-in: dump per-arch intermediate files (.s/.bc/.ll) next to the
        # object files so the generated ISA can be inspected. Build with
        # SAVE_TEMPS=1 and look for *-hip-amdgcn-amd-amdhsa-<arch>.s.
        if os.getenv("SAVE_TEMPS", "0") == "1":
            hipcc_flags += ["--save-temps=obj"]
        # Opt-in (wave32 only): multi-tile backward rasterizer. A 256-thread
        # block (8 wave32 waves) rasterizes 8 separate 8x8 tiles (one per wave)
        # to raise wave occupancy (deep, tile16-like) while keeping the 8x8 tile
        # granularity of the bs32 path. Gated to small CDIM in-source to bound
        # the per-wave register footprint. Build with BS32_MULTITILE=1. Must be
        # defined for BOTH host (launcher grid sizing) and device compilation.
        if os.getenv("BS32_MULTITILE", "0") == "1":
            extra_compile_args["cxx"] += ["-DGSPLAT_BS32_MULTITILE=1"]
            hipcc_flags += ["-DGSPLAT_BS32_MULTITILE=1"]
        # Opt-in: split the multi-tile per-pixel body into a gaussian-test
        # phase and a gradient phase, so both of a lane's two pixels are tested
        # in one EXEC region and their __expf chains can interleave. Device
        # only. Build with BS32_PHASE_SPLIT=1.
        if os.getenv("BS32_PHASE_SPLIT", "0") == "1":
            hipcc_flags += ["-DGSPLAT_BS32_PHASE_SPLIT=1"]
        # Generic extra defines, e.g. GSPLAT_EXTRA_DEFINES="GSPLAT_ATOMIC_CEILING GSPLAT_OPT2"
        for _d in os.getenv("GSPLAT_EXTRA_DEFINES", "").split():
            extra_compile_args["cxx"] += ["-D" + _d]
            hipcc_flags += ["-D" + _d]
        if LINE_INFO:
            hipcc_flags += ["-gline-tables-only"]
        if torch.version.hip:
            # USE_ROCM was added to later versions of PyTorch.
            # Define here to support older PyTorch versions as well:
            define_macros += [("USE_ROCM", "1")]
            undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        if ENABLE_TEST_COVERAGE:
            extra_compile_args['cxx'] += ['-fprofile-instr-generate', '-fcoverage-mapping', '-Qunused-arguments', '--gcc-toolchain=/usr']
            hipcc_flags += ['-fprofile-instr-generate', '-fcoverage-mapping']
            extra_link_args += ['-fprofile-instr-generate']
	# Its still nvcc flags that are used for HIP compilation
        extra_compile_args["nvcc"] = hipcc_flags
        current_dir = pathlib.Path(__file__).parent.resolve()

        include_dirs = [
            osp.join(current_dir, "gsplat", "cuda", "include"),
            f"{os.environ['HOME']}/.local/include",
            f"{os.environ['ROCM_PATH']}/include",
            f"{os.environ['CPLUS_INCLUDE_PATH']}"
        ]

        extension = CUDAExtension(
            # Make sure this matches your package structure
            "gsplat.csrc",  # This changes the extension module name to be more standard
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
        return [extension]
    else:

        from torch.__config__ import parallel_info
        from torch.utils.cpp_extension import CUDAExtension

        extensions_dir = osp.join("gsplat", "cuda")
        sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
            osp.join(extensions_dir, "csrc", "*.cpp")
        )
        sources += [osp.join(extensions_dir, "ext.cpp")]

        undef_macros = []
        define_macros = []

        extra_compile_args = {"cxx": ["-O3"]}
        if not os.name == "nt":  # Not on Windows:
            extra_compile_args["cxx"] += ["-Wno-sign-compare"]
        extra_link_args = [] if WITH_SYMBOLS else ["-s"]

        info = parallel_info()
        if (
            "backend: OpenMP" in info
            and "OpenMP not found" not in info
            and sys.platform != "darwin"
        ):
            extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
            if sys.platform == "win32":
                extra_compile_args["cxx"] += ["/openmp"]
            else:
                extra_compile_args["cxx"] += ["-fopenmp"]
        else:
            print("Compiling without OpenMP...")

        # Compile for mac arm64
        if sys.platform == "darwin" and platform.machine() == "arm64":
            extra_compile_args["cxx"] += ["-arch", "arm64"]
            extra_link_args += ["-arch", "arm64"]

        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]
        if LINE_INFO:
            nvcc_flags += ["-lineinfo"]
        if torch.version.hip:
            # USE_ROCM was added to later versions of PyTorch.
            # Define here to support older PyTorch versions as well:
            define_macros += [("USE_ROCM", None)]
            undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        else:
            nvcc_flags += ["--expt-relaxed-constexpr"]

        # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
        nvcc_flags += ["-diag-suppress", "20012,186"]
        extra_compile_args["nvcc"] = nvcc_flags
        if sys.platform == "win32":
            extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

        current_dir = pathlib.Path(__file__).parent.resolve()
        glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
        include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

        extension = CUDAExtension(
            "gsplat.csrc",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        return [extension]

import torch.utils.cpp_extension as ce

def fixed_get_compiler_abi_compatibility_and_version(compiler):
    try:
        return ce.original_get_compiler_abi_compatibility_and_version(compiler)
    except ValueError:
        # Fallback for clang++ "17.0git"
        return ("gcc", (17, 0))

if not hasattr(ce, "original_get_compiler_abi_compatibility_and_version"):
    ce.original_get_compiler_abi_compatibility_and_version = ce.get_compiler_abi_compatibility_and_version
    ce.get_compiler_abi_compatibility_and_version = fixed_get_compiler_abi_compatibility_and_version


setup(
    name="amd_gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
	author="AMD Corporation",
    license="Apache 2.0",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "twine",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_ext()},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
