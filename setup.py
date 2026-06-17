import glob
import os
import os.path as osp
import pathlib
import platform
import sys
import re

from setuptools import find_packages, setup

IS_ROCM = True
ROCM_HOME = "/opt/rocm"
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
        # gfx1151 fix: honor PYTORCH_ROCM_ARCH first. Upstream get_rocm_arch() runs
        # rocminfo, which finds no GPU at `docker build` time and falls back to
        # gfx942 (CDNA) -> a .so with no gfx1151 code objects (runtime null-deref).
        gpu_arch = (os.environ.get("PYTORCH_ROCM_ARCH") or get_rocm_arch()).split(";")[0].split(",")[0].strip()
        print(f"gpu arch is set to {gpu_arch}")
        conda_prefix = os.getenv("CONDA_PREFIX")
        conda_lib_path = f"{conda_prefix}/lib"
        conda_pip_packages = f"{conda_lib_path}/python3.11/site-packages"

        # Use relative path instead of hardcoded absolute path
        extensions_dir = osp.join("gsplat","cuda")
        sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(osp.join(extensions_dir, "csrc", "*.cpp"))
        sources += [osp.join(extensions_dir, "ext.cpp")]

        undef_macros = []
        define_macros = []

        extra_compile_args = {"cxx": ["-D__HIP_PLATFORM_AMD__" , "-Wno-sign-compare", "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE", "-DUSE_ROCM"]}
        if WITH_SYMBOLS:
            extra_compile_args["cxx"] += ["-g", "-O0"]
        else:
            extra_compile_args = {"cxx": ["-O3", "-Wno-attributes", "-Wno-switch", "-Wno-comment"]}

        extra_link_args = ["-s"]

        # Compile with OpenMP
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        extra_compile_args["cxx"] += ["-fopenmp"]

        hipcc_flags = [ "-D__HIP_PLATFORM_AMD__", "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE", "-DUSE_ROCM" , f"--offload-arch={gpu_arch}"]
        # gfx1151 fix: hipify rewrites the vendored glm's CUDA-version check (CUDA_VERSION ->
        # TORCH_HIP_VERSION) but that macro is undefined when glm/simd/platform.h is parsed,
        # so it evaluates to 0 < 7000 -> #error "GLM requires CUDA 7.0 or higher". Define it
        # high enough to satisfy the (now harmless) hipified version gate.
        hipcc_flags += ["-DTORCH_HIP_VERSION=8000"]
        if WITH_SYMBOLS:
            hipcc_flags += ["-g", "-ggdb" , "-O0"]
        else:
            hipcc_flags += ["-O3" ]
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
            # vendored GLM first — system glm (libglm-dev) is API-incompatible
            osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm"),
            osp.join(current_dir, "gsplat", "cuda", "include"),
            f"{os.environ['HOME']}/.local/include",
            f"/opt/conda/include",
            f"/opt/conda/envs/py_3.12/lib/python3.12/site-packages/",
            f"/opt/rocm/include",
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
            "build",
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
