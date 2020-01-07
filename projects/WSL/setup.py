#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
from os import path
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "wsl", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (
        torch.cuda.is_available() and CUDA_HOME is not None and os.path.isdir(CUDA_HOME)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    # DenseCRF
    crf_sources = glob.glob(os.path.join(extensions_dir, "crf", "densecrf", "src", "*.cpp"))
    crf_include_dirs = os.path.join(extensions_dir, "crf", "densecrf", "include")
    for source in crf_sources:
        if "optimization" in source:
            continue
        if "examples" in source:
            continue
        sources += [source]
    include_dirs += [crf_include_dirs]

    ext_modules = [
        extension(
            "wsl._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="wsl",
    version="0.1",
    author="Yunhang Shen",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=["graphviz "],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
