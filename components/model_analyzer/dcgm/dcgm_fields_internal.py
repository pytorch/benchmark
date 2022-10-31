# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##
# Python bindings for the internal API of DCGM library (dcgm_fields_internal.hpp)
##

from ctypes import *
from ctypes.util import find_library
from . import dcgm_structs

# Provides access to functions
dcgmFP = dcgm_structs._dcgmGetFunctionPointer


#internal-only fields
DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES          = 210 #Memory utilization samples
DCGM_FI_DEV_GPU_UTIL_SAMPLES               = 211 #SM utilization samples
DCGM_FI_DEV_GRAPHICS_PIDS                  = 220 #Graphics processes running on the GPU.
DCGM_FI_DEV_COMPUTE_PIDS                   = 221 #Compute processes running on the GPU.
