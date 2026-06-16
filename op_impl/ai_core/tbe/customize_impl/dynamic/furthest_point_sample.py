#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
FurthestPointSample - AscendC kernel compile & tiling script
Follows the add_custom.py pattern for CANN 8.3RC1 / 310P
"""

import re
import os
import ctypes
import shutil
from tbe.common.platform import get_soc_spec
from tbe.common.utils import para_check
from tbe.tikcpp import compile_op, check_op_cap, get_code_channel, OpInfo
from tbe.tikcpp.compile_op import CommonUtility, AscendCLogLevel
from tbe.common.buildcfg import get_current_build_config
from impl.util.platform_adapter import tbe_register

PYF_PATH = os.path.dirname(os.path.realpath(__file__))

DTYPE_MAP = {
    "float32": ["DT_FLOAT", "float"],
    "float16": ["DT_FLOAT16", "half"],
}


def add_dtype_fmt_option_single(x, x_n, is_ref=False):
    options = []
    x_fmt = x.get("format")
    x_dtype = x.get("dtype")
    x_n_in_kernel = x_n + "_REF" if is_ref else x_n
    options.append(
        "-DDTYPE_{n}={t}".format(n=x_n_in_kernel, t=DTYPE_MAP.get(x_dtype, [None, None])[1])
    )
    options.append(
        "-DORIG_DTYPE_{n}={ot}".format(
            n=x_n_in_kernel, ot=DTYPE_MAP.get(x_dtype, [None, None])[0]
        )
    )
    options.append("-DFORMAT_{n}=FORMAT_{f}".format(n=x_n_in_kernel, f=x_fmt))
    return options


def get_dtype_fmt_options(__inputs__, __outputs__):
    """Generate compile-time dtype/format macros from tensor descriptors."""
    options = []
    input_names = ["points"]
    output_names = ["sampled"]
    unique_param_name_set = set()
    for idx, x in enumerate(__inputs__):
        if x is None:
            continue
        x_n = input_names[idx].upper()
        unique_param_name_set.add(x_n)
        options += add_dtype_fmt_option_single(x, x_n)
    for idx, x in enumerate(__outputs__):
        if x is None:
            continue
        x_n = output_names[idx].upper()
        if x_n in unique_param_name_set:
            options += add_dtype_fmt_option_single(x, x_n, True)
        else:
            options += add_dtype_fmt_option_single(x, x_n)
    return options


def load_dso(so_path):
    try:
        ctypes.CDLL(so_path)
    except OSError as error:
        CommonUtility.print_compile_log("", error, AscendCLogLevel.LOG_ERROR)
        raise RuntimeError("cannot open %s" % (so_path))
    else:
        msg = "load so succ " + so_path
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)


def get_shortsoc_compile_option(compile_option_list: list, shortsoc: str):
    compile_options = []
    if shortsoc in compile_option_list:
        compile_options.extend(compile_option_list[shortsoc])
    if "__ALLSOC__" in compile_option_list:
        compile_options.extend(compile_option_list["__ALLSOC__"])
    return compile_options


def get_kernel_source(src_file, dir_snake, dir_ex):
    """Locate kernel source file using standard search paths."""
    src_ex = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, src_file)
    if os.path.exists(src_ex):
        return src_ex
    src = os.environ.get("BUILD_KERNEL_SRC")
    if src and os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, src_file)
    if os.path.exists(src):
        return src
    return src_ex


def _build_args(points_in, sampled_out):
    """Normalize input/output tensor descriptors."""
    __inputs__ = []
    for arg in [points_in]:
        if arg is not None:
            if isinstance(arg, (list, tuple)):
                if len(arg) > 0:
                    __inputs__.append(arg[0])
                else:
                    continue
            else:
                __inputs__.append(arg)
        else:
            __inputs__.append(arg)
    __outputs__ = []
    for arg in [sampled_out]:
        if arg is not None:
            if isinstance(arg, (list, tuple)):
                if len(arg) > 0:
                    __outputs__.append(arg[0])
                else:
                    continue
            else:
                __outputs__.append(arg)
        else:
            __outputs__.append(arg)
    __attrs__ = []
    return __inputs__, __outputs__, __attrs__


@tbe_register.register_operator("FurthestPointSample", trans_bool_to_s8=False)
@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_INT,
    para_check.KERNEL_NAME,
)
def furthest_point_sample(
    points_in, sampled_out, m, kernel_name="furthest_point_sample", impl_mode=""
):
    """Compile AscendC kernel for FurthestPointSample."""
    if get_current_build_config("enable_op_prebuild"):
        return

    __inputs__, __outputs__, __attrs__ = _build_args(points_in, sampled_out)
    options = get_dtype_fmt_options(__inputs__, __outputs__)
    options += ["-x", "cce"]

    # Resolve bisheng compiler path
    bisheng = os.environ.get("BISHENG_REAL_PATH")
    if bisheng is None:
        bisheng = shutil.which("bisheng")
    if bisheng is not None:
        bisheng_path = os.path.dirname(bisheng)
        tikcpp_path = os.path.realpath(
            os.path.join(bisheng_path, "..", "..", "tikcpp")
        )
    else:
        tikcpp_path = os.path.realpath(
            "/usr/local/Ascend/latest/compiler/tikcpp"
        )

    # AscendC header include paths
    options.append("-I" + tikcpp_path)
    options.append("-I" + os.path.join(tikcpp_path, "..", "..", "include"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    options.append("-I" + os.path.join(tikcpp_path, "..", "ascendc", "act"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common"))
    options.append("-I" + PYF_PATH)

    # Impl mode
    if impl_mode == "high_performance":
        options.append("-DHIGH_PERFORMANCE=1")
    elif impl_mode == "high_precision":
        options.append("-DHIGH_PRECISION=1")

    # Deterministic mode
    if get_current_build_config("enable_deterministic_mode") == 1:
        options.append("-DDETERMINISTIC_MODE=1")
    else:
        options.append("-DDETERMINISTIC_MODE=0")

    # AscendC API version
    ascendc_api_version_path = os.path.join(
        tikcpp_path, "tikcfw/lib/ascendc_api_version.h"
    )
    if os.path.exists(ascendc_api_version_path):
        with open(ascendc_api_version_path, "r") as f:
            ver = re.findall(r"#define ASCENDC_API_VERSION (\d+)", f.read())
            if ver:
                options.append(f"-DASCENDC_API_VERSION={ver[0]}")

    # Soci-specific compile options
    custom_compile_options = {
        "__ALLSOC__": [
            "--cce-auto-sync=on",
            "-Wno-deprecated-declarations",
            "-Werror",
        ]
    }
    soc_short = get_soc_spec("SHORT_SOC_VERSION").lower()
    compile_opts = get_shortsoc_compile_option(custom_compile_options, soc_short)
    options += compile_opts

    # Kernel source location
    origin_func_name = "furthest_point_sample"
    src = get_kernel_source(
        "furthest_point_sample_kernel.cpp",
        "furthest_point_sample",
        "furthest_point_sample",
    )

    msg = (
        "start compile Ascend C Operator FurthestPointSample, kernel name is "
        + kernel_name
    )
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)

    op_type = "FurthestPointSample"
    code_channel = get_code_channel(src, kernel_name, op_type, options)
    op_info = OpInfo(
        kernel_name=kernel_name,
        op_type=op_type,
        inputs=__inputs__,
        outputs=__outputs__,
        attrs=__attrs__,
        impl_mode=impl_mode,
        origin_inputs=[points_in],
        origin_outputs=[sampled_out],
        param_type_dynamic=False,
        mc2_ctx=[],
        param_type_list=["required", "required", "required"],
        init_value_list=[None],
        output_shape_depend_on_compute=[],
    )
    compile_op(
        src,
        origin_func_name,
        op_info,
        options,
        code_channel,
        "{}",
        {"valueDepend": {}},
    )


def op_select_format(points_in, sampled_out, impl_mode=""):
    __inputs__, __outputs__, __attrs__ = _build_args(points_in, sampled_out)
    result = check_op_cap(
        "op_select_format", "FurthestPointSample",
        __inputs__, __outputs__, __attrs__
    )
    return result.decode("utf-8")


def get_op_specific_info(points_in, sampled_out, impl_mode=""):
    __inputs__, __outputs__, __attrs__ = _build_args(points_in, sampled_out)
    result = check_op_cap(
        "get_op_specific_info", "FurthestPointSample",
        __inputs__, __outputs__, __attrs__
    )
    return result.decode("utf-8")
