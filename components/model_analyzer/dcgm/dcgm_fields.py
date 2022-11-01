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
# Python bindings for the internal API of DCGM library (dcgm_fields.h)
##

from ctypes import *
from ctypes.util import find_library
from . import dcgm_structs

# Provides access to functions
dcgmFP = dcgm_structs._dcgmGetFunctionPointer

# Field Types are a single byte. List these in ASCII order
DCGM_FT_BINARY      = 'b' # Blob of binary data representing a structure
DCGM_FT_DOUBLE      = 'd' # 8-byte double precision 
DCGM_FT_INT64       = 'i' # 8-byte signed integer 
DCGM_FT_STRING      = 's' # Null-terminated ASCII Character string 
DCGM_FT_TIMESTAMP   = 't' # 8-byte signed integer usec since 1970 

# Field scope. What are these fields associated with
DCGM_FS_GLOBAL	    = 0              # Field is global (ex: driver version)
DCGM_FS_ENTITY      = 1              # Field is associated with an entity (GPU, VGPU, ..etc)
DCGM_FS_DEVICE      = DCGM_FS_ENTITY # Field is associated with a device. Deprecated. Use DCGM_FS_ENTITY

# DCGM_FI_DEV_CLOCK_THROTTLE_REASONS is a bitmap of why the clock is throttled.
# These macros are masks for relevant throttling, and are a 1:1 map to the NVML
# reasons documented in nvml.h. The notes for the header are copied blow:

# Nothing is running on the GPU and the clocks are dropping to Idle state
DCGM_CLOCKS_THROTTLE_REASON_GPU_IDLE = 0x0000000000000001

# GPU clocks are limited by current setting of applications clocks
DCGM_CLOCKS_THROTTLE_REASON_CLOCKS_SETTING = 0x0000000000000002

# SW Power Scaling algorithm is reducing the clocks below requested clocks 
DCGM_CLOCKS_THROTTLE_REASON_SW_POWER_CAP = 0x0000000000000004

# HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
#
# This is an indicator of:
#  - temperature being too high
#  - External Power Brake Assertion is triggered (e.g. by the system power supply)
#  - Power draw is too high and Fast Trigger protection is reducing the clocks
#  - May be also reported during PState or clock change
#  - This behavior may be removed in a later release.

DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN = 0x0000000000000008

# Sync Boost
#
# This GPU has been added to a Sync boost group with nvidia-smi or DCGM in
# order to maximize performance per watt. All GPUs in the sync boost group
# will boost to the minimum possible clocks across the entire group. Look at
# the throttle reasons for other GPUs in the system to see why those GPUs are
# holding this one at lower clocks.
DCGM_CLOCKS_THROTTLE_REASON_SYNC_BOOST = 0x0000000000000010

# SW Thermal Slowdown
#
# This is an indicator of one or more of the following:
#  - Current GPU temperature above the GPU Max Operating Temperature
#  - Current memory temperature above the Memory Max Operating Temperature
DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL = 0x0000000000000020

# HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
#
# This is an indicator of:
#  - temperature being too high
DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL = 0x0000000000000040

# HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
#
# This is an indicator of:
#  - External Power Brake Assertion being triggered (e.g. by the system power supply)
DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE = 0x0000000000000080

# GPU clocks are limited by current setting of Display clocks
DCGM_CLOCKS_THROTTLE_REASON_DISPLAY_CLOCKS = 0x0000000000000100

#Field entity groups. Which type of entity is this field or field value associated with
DCGM_FE_NONE   = 0 # Field is not associated with an entity. Field scope should be DCGM_FS_GLOBAL
DCGM_FE_GPU    = 1 # Field is associated with a GPU entity
DCGM_FE_VGPU   = 2 # Field is associated with a VGPU entity
DCGM_FE_SWITCH = 3 # Field is associated with a Switch entity
DCGM_FE_GPU_I  = 4 # Field is associated with a GPU Instance entity
DCGM_FE_GPU_CI = 5 # Field is associated with a GPU Compute Instance entity
DCGM_FE_LINK   = 6 # Field is associated with an NVLINK

c_dcgm_field_eid_t = c_uint32 #Represents an identifier for an entity within a field entity. For instance, this is the gpuId for DCGM_FE_GPU.

#System attributes
DCGM_FI_UNKNOWN                 = 0
DCGM_FI_DRIVER_VERSION          = 1   #Driver Version
DCGM_FI_NVML_VERSION            = 2   #Underlying NVML version
DCGM_FI_PROCESS_NAME            = 3   #Process Name. Will be nv-hostengine or your process's name in embedded mode
DCGM_FI_DEV_COUNT               = 4   #Number of Devices on the node
DCGM_FI_CUDA_DRIVER_VERSION     = 5   #Cuda Driver Version as an integer. CUDA 11.1 = 11100
#Device attributes
DCGM_FI_DEV_NAME                = 50  #Name of the GPU device
DCGM_FI_DEV_BRAND               = 51  #Device Brand
DCGM_FI_DEV_NVML_INDEX          = 52  #NVML index of this GPU
DCGM_FI_DEV_SERIAL              = 53  #Device Serial Number
DCGM_FI_DEV_UUID                = 54  #UUID corresponding to the device
DCGM_FI_DEV_MINOR_NUMBER        = 55  #Device node minor number /dev/nvidia#
DCGM_FI_DEV_OEM_INFOROM_VER     = 56  #OEM inforom version
DCGM_FI_DEV_PCI_BUSID           = 57  #PCI attributes for the device
DCGM_FI_DEV_PCI_COMBINED_ID     = 58  #The combined 16-bit device id and 16-bit vendor id
DCGM_FI_DEV_PCI_SUBSYS_ID       = 59  #The 32-bit Sub System Device ID
DCGM_FI_GPU_TOPOLOGY_PCI        = 60  #Topology of all GPUs on the system via PCI (static)
DCGM_FI_GPU_TOPOLOGY_NVLINK     = 61  #Topology of all GPUs on the system via NVLINK (static)
DCGM_FI_GPU_TOPOLOGY_AFFINITY   = 62  #Affinity of all GPUs on the system (static)
DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY = 63 #Cuda compute capability for the device
DCGM_FI_DEV_COMPUTE_MODE        = 65  #Compute mode for the device
DCGM_FI_DEV_PERSISTENCE_MODE    = 66  #Persistence mode for the device
DCGM_FI_DEV_MIG_MODE            = 67  #MIG mode for the device
DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR = 68  #String value for CUDA_VISIBLE_DEVICES for the device
DCGM_FI_DEV_MIG_MAX_SLICES      = 69  #The maximum number of slices this GPU supports
DCGM_FI_DEV_CPU_AFFINITY_0      = 70  #Device CPU affinity. part 1/8 = cpus 0 - 63
DCGM_FI_DEV_CPU_AFFINITY_1      = 71  #Device CPU affinity. part 1/8 = cpus 64 - 127
DCGM_FI_DEV_CPU_AFFINITY_2      = 72  #Device CPU affinity. part 2/8 = cpus 128 - 191
DCGM_FI_DEV_CPU_AFFINITY_3      = 73  #Device CPU affinity. part 3/8 = cpus 192 - 255
DCGM_FI_DEV_CC_MODE             = 74  #Device CC/APM mode
DCGM_FI_DEV_MIG_ATTRIBUTES      = 75  #MIG device attributes
DCGM_FI_DEV_MIG_GI_INFO         = 76  #GPU instance profile information
DCGM_FI_DEV_MIG_CI_INFO         = 77  #Compute instance profile information
DCGM_FI_DEV_ECC_INFOROM_VER     = 80  #ECC inforom version
DCGM_FI_DEV_POWER_INFOROM_VER   = 81  #Power management object inforom version
DCGM_FI_DEV_INFOROM_IMAGE_VER   = 82  #Inforom image version
DCGM_FI_DEV_INFOROM_CONFIG_CHECK = 83 #Inforom configuration checksum
DCGM_FI_DEV_INFOROM_CONFIG_VALID = 84 #Reads the infoROM from the flash and verifies the checksums
DCGM_FI_DEV_VBIOS_VERSION       = 85  #VBIOS version of the device
DCGM_FI_DEV_BAR1_TOTAL          = 90  #Total BAR1 of the GPU
DCGM_FI_SYNC_BOOST              = 91  #Deprecated - Sync boost settings on the node
DCGM_FI_DEV_BAR1_USED           = 92  #Used BAR1 of the GPU in MB
DCGM_FI_DEV_BAR1_FREE           = 93  #Free BAR1 of the GPU in MB
#Clocks and power
DCGM_FI_DEV_SM_CLOCK            = 100 #SM clock for the device
DCGM_FI_DEV_MEM_CLOCK           = 101 #Memory clock for the device
DCGM_FI_DEV_VIDEO_CLOCK         = 102 #Video encoder/decoder clock for the device
DCGM_FI_DEV_APP_SM_CLOCK        = 110 #SM Application clocks
DCGM_FI_DEV_APP_MEM_CLOCK       = 111 #Memory Application clocks
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS = 112 #Current clock throttle reasons (bitmask of DCGM_CLOCKS_THROTTLE_REASON_*)
DCGM_FI_DEV_MAX_SM_CLOCK        = 113 #Maximum supported SM clock for the device
DCGM_FI_DEV_MAX_MEM_CLOCK       = 114 #Maximum supported Memory clock for the device
DCGM_FI_DEV_MAX_VIDEO_CLOCK     = 115 #Maximum supported Video encoder/decoder clock for the device
DCGM_FI_DEV_AUTOBOOST           = 120 #Auto-boost for the device (1 = enabled. 0 = disabled)
DCGM_FI_DEV_SUPPORTED_CLOCKS    = 130 #Supported clocks for the device
DCGM_FI_DEV_MEMORY_TEMP         = 140 #Memory temperature for the device
DCGM_FI_DEV_GPU_TEMP            = 150 #Current temperature readings for the device, in degrees C
DCGM_FI_DEV_MEM_MAX_OP_TEMP     = 151 #Maximum operating temperature for the memory of this GPU
DCGM_FI_DEV_GPU_MAX_OP_TEMP     = 152 #Maximum operating temperature for this GPU
DCGM_FI_DEV_POWER_USAGE         = 155 #Power usage for the device in Watts
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 156 #Total energy consumption for the GPU in mJ since the driver was last reloaded
DCGM_FI_DEV_SLOWDOWN_TEMP       = 158 #Slowdown temperature for the device
DCGM_FI_DEV_SHUTDOWN_TEMP       = 159 #Shutdown temperature for the device
DCGM_FI_DEV_POWER_MGMT_LIMIT    = 160 #Current Power limit for the device
DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN = 161 #Minimum power management limit for the device
DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX = 162 #Maximum power management limit for the device
DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF = 163 #Default power management limit for the device
DCGM_FI_DEV_ENFORCED_POWER_LIMIT = 164 #Effective power limit that the driver enforces after taking into account all limiters
DCGM_FI_DEV_PSTATE               = 190 #Performance state (P-State) 0-15. 0=highest
DCGM_FI_DEV_FAN_SPEED            = 191 #Fan speed for the device in percent 0-100
#Device utilization and telemetry
DCGM_FI_DEV_PCIE_TX_THROUGHPUT  = 200 #Deprecated - PCIe Tx utilization information
DCGM_FI_DEV_PCIE_RX_THROUGHPUT  = 201 #Deprecated - PCIe Rx utilization information
DCGM_FI_DEV_PCIE_REPLAY_COUNTER = 202 #PCIe replay counter
DCGM_FI_DEV_GPU_UTIL            = 203 #GPU Utilization
DCGM_FI_DEV_MEM_COPY_UTIL       = 204 #Memory Utilization
DCGM_FI_DEV_ACCOUNTING_DATA     = 205 #Process accounting stats
DCGM_FI_DEV_ENC_UTIL            = 206 #Encoder utilization
DCGM_FI_DEV_DEC_UTIL            = 207 #Decoder utilization
# Fields 210, 211, 220, and 221 are internal-only. see dcgm_fields_internal.py 
DCGM_FI_DEV_XID_ERRORS          = 230 #XID errors. The value is the specific XID error
DCGM_FI_DEV_PCIE_MAX_LINK_GEN   = 235 #PCIe Max Link Generation
DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH = 236 #PCIe Max Link Width
DCGM_FI_DEV_PCIE_LINK_GEN       = 237 #PCIe Current Link Generation
DCGM_FI_DEV_PCIE_LINK_WIDTH     = 238 #PCIe Current Link Width
#Violation counters
DCGM_FI_DEV_POWER_VIOLATION     = 240 #Power Violation time in usec
DCGM_FI_DEV_THERMAL_VIOLATION   = 241 #Thermal Violation time in usec
DCGM_FI_DEV_SYNC_BOOST_VIOLATION = 242 #Sync Boost Violation time in usec
DCGM_FI_DEV_BOARD_LIMIT_VIOLATION = 243 #Board Limit Violation time in usec.
DCGM_FI_DEV_LOW_UTIL_VIOLATION    = 244 #Low Utilization Violation time in usec.
DCGM_FI_DEV_RELIABILITY_VIOLATION = 245 #Reliability Violation time in usec.
DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION = 246 #App Clocks Violation time in usec.
DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION = 247 #Base Clocks Violation time in usec.
#Framebuffer usage
DCGM_FI_DEV_FB_TOTAL            = 250 #Total framebuffer memory in MB
DCGM_FI_DEV_FB_FREE             = 251 #Total framebuffer used in MB
DCGM_FI_DEV_FB_USED             = 252 #Total framebuffer free in MB
DCGM_FI_DEV_FB_RESERVED         = 253 #Total framebuffer reserved in MB
#Device ECC Counters
DCGM_FI_DEV_ECC_CURRENT         = 300 #Current ECC mode for the device
DCGM_FI_DEV_ECC_PENDING         = 301 #Pending ECC mode for the device
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL   = 310 #Total single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL   = 311 #Total double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_TOTAL   = 312 #Total single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_TOTAL   = 313 #Total double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_L1      = 314 #L1 cache single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_L1      = 315 #L1 cache double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_L2      = 316 #L2 cache single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_L2      = 317 #L2 cache double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_DEV     = 318 #Device memory single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_DEV     = 319 #Device memory double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_REG     = 320 #Register file single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_REG     = 321 #Register file double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_TEX     = 322 #Texture memory single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_TEX     = 323 #Texture memory double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_L1      = 324 #L1 cache single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_L1      = 325 #L1 cache double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_L2      = 326 #L2 cache single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_L2      = 327 #L2 cache double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_DEV     = 328 #Device memory single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_DEV     = 329 #Device memory double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_REG     = 330 #Register File single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_REG     = 331 #Register File double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_TEX     = 332 #Texture memory single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_TEX     = 333 #Texture memory double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_RETIRED_SBE         = 390 #Number of retired pages because of single bit errors
DCGM_FI_DEV_RETIRED_DBE         = 391 #Number of retired pages because of double bit errors
DCGM_FI_DEV_RETIRED_PENDING     = 392 #Number of pages pending retirement
#Row remapper fields (Ampere and newer)
DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS = 393 #Number of remapped rows for uncorrectable errors
DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS   = 394 #Number of remapped rows for correctable errors
DCGM_FI_DEV_ROW_REMAP_FAILURE           = 395 #Whether remapping of rows has failed
DCGM_FI_DEV_ROW_REMAP_PENDING           = 396 #Whether remapping of rows is pending

#Device NvLink Bandwidth and Error Counters
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0 = 400 #NV Link flow control CRC  Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1 = 401 #NV Link flow control CRC  Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2 = 402 #NV Link flow control CRC  Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3 = 403 #NV Link flow control CRC  Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4 = 404 #NV Link flow control CRC  Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5 = 405 #NV Link flow control CRC  Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL = 409 #NV Link flow control CRC  Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0 = 410 #NV Link data CRC Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1 = 411 #NV Link data CRC Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2 = 412 #NV Link data CRC Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3 = 413 #NV Link data CRC Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4 = 414 #NV Link data CRC Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5 = 415 #NV Link data CRC Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL = 419 #NV Link data CRC Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0   = 420 #NV Link Replay Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1   = 421 #NV Link Replay Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2   = 422 #NV Link Replay Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3   = 423 #NV Link Replay Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4   = 424 #NV Link Replay Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5   = 425 #NV Link Replay Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL = 429 #NV Link Replay Error Counter total for all Lanes

DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0 = 430 #NV Link Recovery Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1 = 431 #NV Link Recovery Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2 = 432 #NV Link Recovery Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3 = 433 #NV Link Recovery Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4 = 434 #NV Link Recovery Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5 = 435 #NV Link Recovery Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL = 439 #NV Link Recovery Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_BANDWIDTH_L0            = 440 #NV Link Bandwidth Counter for Lane 0
DCGM_FI_DEV_NVLINK_BANDWIDTH_L1            = 441 #NV Link Bandwidth Counter for Lane 1
DCGM_FI_DEV_NVLINK_BANDWIDTH_L2            = 442 #NV Link Bandwidth Counter for Lane 2
DCGM_FI_DEV_NVLINK_BANDWIDTH_L3            = 443 #NV Link Bandwidth Counter for Lane 3
DCGM_FI_DEV_NVLINK_BANDWIDTH_L4            = 444 #NV Link Bandwidth Counter for Lane 4
DCGM_FI_DEV_NVLINK_BANDWIDTH_L5            = 445 #NV Link Bandwidth Counter for Lane 5
DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL         = 449 #NV Link Bandwidth Counter total for all Lanes
DCGM_FI_DEV_GPU_NVLINK_ERRORS              = 450 #GPU NVLink error information
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6  = 451
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7  = 452
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8  = 453
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9  = 454
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10 = 455
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11 = 456
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12 = 406
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13 = 407
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14 = 408
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15 = 481
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16 = 482
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17 = 483
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6  = 457
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7  = 458
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8  = 459
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9  = 460
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10 = 461
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11 = 462
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12 = 416
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13 = 417
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14 = 418
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15 = 484
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16 = 485
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17 = 486
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6    = 463
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7    = 464
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8    = 465
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9    = 466
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10   = 467
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11   = 468
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12   = 426
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13   = 427
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14   = 428
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15   = 487
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16   = 488
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17   = 489
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6  = 469
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7  = 470
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8  = 471
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9  = 472
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10 = 473
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11 = 474
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12 = 436
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13 = 437
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14 = 438
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15 = 491
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16 = 492
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17 = 493
DCGM_FI_DEV_NVLINK_BANDWIDTH_L6  = 475
DCGM_FI_DEV_NVLINK_BANDWIDTH_L7  = 476
DCGM_FI_DEV_NVLINK_BANDWIDTH_L8  = 477
DCGM_FI_DEV_NVLINK_BANDWIDTH_L9  = 478
DCGM_FI_DEV_NVLINK_BANDWIDTH_L10 = 479
DCGM_FI_DEV_NVLINK_BANDWIDTH_L11 = 480
DCGM_FI_DEV_NVLINK_BANDWIDTH_L12 = 446
DCGM_FI_DEV_NVLINK_BANDWIDTH_L13 = 447
DCGM_FI_DEV_NVLINK_BANDWIDTH_L14 = 448
DCGM_FI_DEV_NVLINK_BANDWIDTH_L15 = 494
DCGM_FI_DEV_NVLINK_BANDWIDTH_L16 = 495
DCGM_FI_DEV_NVLINK_BANDWIDTH_L17 = 496

#Device Attributes associated with virtualization
DCGM_FI_DEV_VIRTUAL_MODE                   = 500  #Operating mode of the GPU
DCGM_FI_DEV_SUPPORTED_TYPE_INFO            = 501  #Includes Count and Supported vGPU type information
DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS        = 502  #Includes Count and List of Creatable vGPU type IDs
DCGM_FI_DEV_VGPU_INSTANCE_IDS              = 503  #Includes Count and List of vGPU instance IDs
DCGM_FI_DEV_VGPU_UTILIZATIONS              = 504  #Utilization values for vGPUs running on the device
DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION   = 505  #Utilization values for processes running within vGPU VMs using the device
DCGM_FI_DEV_ENC_STATS                      = 506  #Current encoder statistics for a given device
DCGM_FI_DEV_FBC_STATS                      = 507  #Statistics of current active frame buffer capture sessions on a given device
DCGM_FI_DEV_FBC_SESSIONS_INFO              = 508  #Information about active frame buffer capture sessions on a target device
DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS        = 509  #Includes Count and currently Supported vGPU types on a device
DCGM_FI_DEV_VGPU_TYPE_INFO                 = 510  #Includes Static info of vGPU types supported on a device
DCGM_FI_DEV_VGPU_TYPE_NAME                 = 511  #Includes the name of a vGPU type supported on a device
DCGM_FI_DEV_VGPU_TYPE_CLASS                = 512  #Includes the class of a vGPU type supported on a device
DCGM_FI_DEV_VGPU_TYPE_LICENSE              = 513  #Includes the license info for a vGPU type supported on a device
#Related to vGPU Instance IDs
DCGM_FI_DEV_VGPU_VM_ID                   = 520  #vGPU VM ID
DCGM_FI_DEV_VGPU_VM_NAME                 = 521  #vGPU VM name
DCGM_FI_DEV_VGPU_TYPE                    = 522  #vGPU type of the vGPU instance
DCGM_FI_DEV_VGPU_UUID                    = 523  #UUID of the vGPU instance
DCGM_FI_DEV_VGPU_DRIVER_VERSION          = 524  #Driver version of the vGPU instance
DCGM_FI_DEV_VGPU_MEMORY_USAGE            = 525  #Memory usage of the vGPU instance
DCGM_FI_DEV_VGPU_LICENSE_STATUS          = 526  #License status of the vGPU
DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT        = 527  #Frame rate limit of the vGPU instance
DCGM_FI_DEV_VGPU_ENC_STATS               = 528  #Current encoder statistics of the vGPU instance
DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO       = 529  #Information about all active encoder sessions on the vGPU instance
DCGM_FI_DEV_VGPU_FBC_STATS               = 530  #Statistics of current active frame buffer capture sessions on the vGPU instance
DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO       = 531  #Information about active frame buffer capture sessions on the vGPU instance
DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE  = 532  #License state information of the vGPU instance
DCGM_FI_DEV_VGPU_PCI_ID                  = 533  #PCI Id of the vGPU instance
DCGM_FI_DEV_VGPU_VM_GPU_INSTANCE_ID      = 534  #GPU Instance Id of the vGPU instance
#Internal fields reserve the range 600..699
#below fields related to NVSwitch
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00             = 700
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P00             = 701
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P00            = 702
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P00             = 703
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P01             = 704
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P01             = 705
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P01            = 706
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P01             = 707
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P02             = 708
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P02             = 709
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P02            = 710
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P02             = 711
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P03             = 712
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P03             = 713
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P03            = 714
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P03             = 715
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P04             = 716
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P04             = 717
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P04            = 718
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P04             = 719
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P05             = 720
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P05             = 721
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P05            = 722
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P05             = 723
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P06             = 724
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P06             = 725
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P06            = 726
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P06             = 727
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P07             = 728
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P07             = 729
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P07            = 730
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P07             = 731
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P08             = 732
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P08             = 733
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P08            = 734
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P08             = 735
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P09             = 736
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P09             = 737
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P09            = 738
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P09             = 739
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P10             = 740
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P10             = 741
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P10            = 742
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P10             = 743
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P11             = 744
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P11             = 745
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P11            = 746
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P11             = 747
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P12             = 748
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P12             = 749
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P12            = 750
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P12             = 751
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P13             = 752
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P13             = 753
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P13            = 754
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P13             = 755
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P14             = 756
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P14             = 757
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P14            = 758
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P14             = 759
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P15             = 760
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P15             = 761
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P15            = 762
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P15             = 763
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P16             = 764
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P16             = 765
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P16            = 766
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P16             = 767
DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P17             = 768
DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P17             = 769
DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P17            = 770
DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P17             = 771
DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX          = 780
DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX          = 781
DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS           = 782
DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS       = 783
DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS          = 784
DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS        = 785
DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS            = 786
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS             = 787
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS             = 788
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0        = 789
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1        = 790
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2        = 791
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3        = 792
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0     = 793
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1     = 794
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2     = 795
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3     = 796
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0       = 797
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1       = 798
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2       = 799
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3       = 800
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0      = 801
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1      = 802
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2      = 803
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3      = 804
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0      = 805
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1      = 806
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2      = 807
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3      = 808
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0       = 809
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1       = 810
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2       = 811
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3       = 812
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0       = 813
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1       = 814
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2       = 815
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3       = 816
DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS                = 856
DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS            = 857
DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT         = 858
DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN  = 859
DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN  = 860
DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX               = 861
DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX               = 862

'''
Profiling Fields
'''
DCGM_FI_PROF_GR_ENGINE_ACTIVE                    = 1001 #Ratio of time the graphics engine is active. The graphics engine is 
                                                        #active if a graphics/compute context is bound and the graphics pipe or 
                                                        #compute pipe is busy.

DCGM_FI_PROF_SM_ACTIVE                           = 1002 #The ratio of cycles an SM has at least 1 warp assigned 
                                                        #(computed from the number of cycles and elapsed cycles) 

DCGM_FI_PROF_SM_OCCUPANCY                        = 1003 #The ratio of number of warps resident on an SM. 
                                                        #(number of resident as a ratio of the theoretical 
                                                        #maximum number of warps per elapsed cycle)

DCGM_FI_PROF_PIPE_TENSOR_ACTIVE                  = 1004 #The ratio of cycles the any tensor pipe is active
                                                        #(off the peak sustained elapsed cycles)

DCGM_FI_PROF_DRAM_ACTIVE                         = 1005 #The ratio of cycles the device memory interface is active sending or receiving data.
DCGM_FI_PROF_PIPE_FP64_ACTIVE                    = 1006 #Ratio of cycles the fp64 pipe is active.
DCGM_FI_PROF_PIPE_FP32_ACTIVE                    = 1007 #Ratio of cycles the fp32 pipe is active.
DCGM_FI_PROF_PIPE_FP16_ACTIVE                    = 1008 #Ratio of cycles the fp16 pipe is active. This does not include HMMA.
DCGM_FI_PROF_PCIE_TX_BYTES                       = 1009 #The number of bytes of active PCIe tx (transmit) data including both header and payload.
DCGM_FI_PROF_PCIE_RX_BYTES                       = 1010 #The number of bytes of active PCIe rx (read) data including both header and payload.
DCGM_FI_PROF_NVLINK_TX_BYTES                     = 1011 #The number of bytes of active NvLink tx (transmit) data including both header and payload.
DCGM_FI_PROF_NVLINK_RX_BYTES                     = 1012 #The number of bytes of active NvLink rx (receive) data including both header and payload.
DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE             = 1013 #The ratio of cycles the IMMA tensor pipe is active (off the peak sustained elapsed cycles)
DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE             = 1014 #The ratio of cycles the HMMA tensor pipe is active (off the peak sustained elapsed cycles)
DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE             = 1015 #The ratio of cycles the tensor (DFMA) pipe is active (off the peak sustained elapsed cycles)
DCGM_FI_PROF_PIPE_INT_ACTIVE                     = 1016 #Ratio of cycles the integer pipe is active.

#Ratio of cycles each of the NVDEC engines are active.
DCGM_FI_PROF_NVDEC0_ACTIVE = 1017
DCGM_FI_PROF_NVDEC1_ACTIVE = 1018
DCGM_FI_PROF_NVDEC2_ACTIVE = 1019
DCGM_FI_PROF_NVDEC3_ACTIVE = 1020
DCGM_FI_PROF_NVDEC4_ACTIVE = 1021
DCGM_FI_PROF_NVDEC5_ACTIVE = 1022
DCGM_FI_PROF_NVDEC6_ACTIVE = 1023
DCGM_FI_PROF_NVDEC7_ACTIVE = 1024

#Ratio of cycles each of the NVJPG engines are active.
DCGM_FI_PROF_NVJPG0_ACTIVE = 1025
DCGM_FI_PROF_NVJPG1_ACTIVE = 1026
DCGM_FI_PROF_NVJPG2_ACTIVE = 1027
DCGM_FI_PROF_NVJPG3_ACTIVE = 1028
DCGM_FI_PROF_NVJPG4_ACTIVE = 1029
DCGM_FI_PROF_NVJPG5_ACTIVE = 1030
DCGM_FI_PROF_NVJPG6_ACTIVE = 1031
DCGM_FI_PROF_NVJPG7_ACTIVE = 1032

#Ratio of cycles each of the NVOFA engines are active.
DCGM_FI_PROF_NVOFA0_ACTIVE = 1033

'''
The per-link number of bytes of active NvLink TX (transmit) or RX (transmit) data including both header and payload.
For example: DCGM_FI_PROF_NVLINK_L0_TX_BYTES -> L0 TX
To get the bandwidth for a link, add the RX and TX value together like 
total = DCGM_FI_PROF_NVLINK_L0_TX_BYTES + DCGM_FI_PROF_NVLINK_L0_RX_BYTES
'''
DCGM_FI_PROF_NVLINK_L0_TX_BYTES  = 1040
DCGM_FI_PROF_NVLINK_L0_RX_BYTES  = 1041
DCGM_FI_PROF_NVLINK_L1_TX_BYTES  = 1042
DCGM_FI_PROF_NVLINK_L1_RX_BYTES  = 1043
DCGM_FI_PROF_NVLINK_L2_TX_BYTES  = 1044
DCGM_FI_PROF_NVLINK_L2_RX_BYTES  = 1045
DCGM_FI_PROF_NVLINK_L3_TX_BYTES  = 1046
DCGM_FI_PROF_NVLINK_L3_RX_BYTES  = 1047
DCGM_FI_PROF_NVLINK_L4_TX_BYTES  = 1048
DCGM_FI_PROF_NVLINK_L4_RX_BYTES  = 1049
DCGM_FI_PROF_NVLINK_L5_TX_BYTES  = 1050
DCGM_FI_PROF_NVLINK_L5_RX_BYTES  = 1051
DCGM_FI_PROF_NVLINK_L6_TX_BYTES  = 1052
DCGM_FI_PROF_NVLINK_L6_RX_BYTES  = 1053
DCGM_FI_PROF_NVLINK_L7_TX_BYTES  = 1054
DCGM_FI_PROF_NVLINK_L7_RX_BYTES  = 1055
DCGM_FI_PROF_NVLINK_L8_TX_BYTES  = 1056
DCGM_FI_PROF_NVLINK_L8_RX_BYTES  = 1057
DCGM_FI_PROF_NVLINK_L9_TX_BYTES  = 1058
DCGM_FI_PROF_NVLINK_L9_RX_BYTES  = 1059
DCGM_FI_PROF_NVLINK_L10_TX_BYTES = 1060
DCGM_FI_PROF_NVLINK_L10_RX_BYTES = 1061
DCGM_FI_PROF_NVLINK_L11_TX_BYTES = 1062
DCGM_FI_PROF_NVLINK_L11_RX_BYTES = 1063
DCGM_FI_PROF_NVLINK_L12_TX_BYTES = 1064
DCGM_FI_PROF_NVLINK_L12_RX_BYTES = 1065
DCGM_FI_PROF_NVLINK_L13_TX_BYTES = 1066
DCGM_FI_PROF_NVLINK_L13_RX_BYTES = 1067
DCGM_FI_PROF_NVLINK_L14_TX_BYTES = 1068
DCGM_FI_PROF_NVLINK_L14_RX_BYTES = 1069
DCGM_FI_PROF_NVLINK_L15_TX_BYTES = 1070
DCGM_FI_PROF_NVLINK_L15_RX_BYTES = 1071
DCGM_FI_PROF_NVLINK_L16_TX_BYTES = 1072
DCGM_FI_PROF_NVLINK_L16_RX_BYTES = 1073
DCGM_FI_PROF_NVLINK_L17_TX_BYTES = 1074
DCGM_FI_PROF_NVLINK_L17_RX_BYTES = 1075

DCGM_FI_PROF_NVLINK_THROUGHPUT_FIRST = DCGM_FI_PROF_NVLINK_L0_TX_BYTES
DCGM_FI_PROF_NVLINK_THROUGHPUT_LAST  = DCGM_FI_PROF_NVLINK_L17_RX_BYTES

#greater than maximum fields above. This value can increase in the future
DCGM_FI_MAX_FIELDS              = 1076


class struct_c_dcgm_field_meta_t(dcgm_structs._DcgmStructure):
    # struct_c_dcgm_field_meta_t structure
    pass # opaque handle
dcgm_field_meta_t = POINTER(struct_c_dcgm_field_meta_t)


class _PrintableStructure(dcgm_structs._DcgmStructure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >>> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}

    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = "%s"
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif "<default>" in self._fmt_:
                fmt = self._fmt_["<default>"]
            result.append(("%s: " + fmt) % (key, value))
        return self.__class__.__name__ + "(" + ', '.join(result) + ")"


# Provides access to functions from dcgm_agent_internal
dcgmFP = dcgm_structs._dcgmGetFunctionPointer

SHORTNAME_LENGTH = 10
UNIT_LENGTH = 4

# Structure to hold formatting information for values
class c_dcgm_field_output_format_t(_PrintableStructure):
    _fields_ = [
        ('shortName', c_char * SHORTNAME_LENGTH),
        ('unit' , c_char * UNIT_LENGTH),
        ('width' , c_short)
    ]

TAG_LENGTH = 48

# Structure to represent device information
class c_dcgm_field_meta_t(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('fieldId', c_short),
        ('fieldType', c_char),
        ('size', c_ubyte),
        ('tag', c_char * TAG_LENGTH),
        ('scope', c_int),
        ('valueFormat', c_dcgm_field_output_format_t)
    ]


# Class for maintaining properties for each sampling type like Power, Utilization and Clock.
class pySamplingProperties:
    '''
    The instance of this class is used to hold information related to each sampling event type.
    '''

    def __init__(self, name, sampling_type, sample_val_type, timeIntervalIdle, timeIntervalBoost, min_value, max_value):
        self.name = name
        self.sampling_type = sampling_type
        self.timeIntervalIdle = timeIntervalIdle
        self.timeIntervalBoost = timeIntervalBoost
        self.min_value = min_value
        self.max_value = max_value
        self.sample_val_type = sample_val_type

def DcgmFieldsInit():
    fn = dcgmFP("DcgmFieldsInit")
    ret = fn()
    assert ret == 0, "Got return %d from DcgmFieldsInit" % ret

def DcgmFieldGetById(fieldId):
    '''
    Get metadata for a field, given its fieldId

    :param fieldId: Field ID to get metadata for
    :return: c_dcgm_field_meta_t struct on success. None on error.
    '''
    DcgmFieldsInit()

    retVal = c_dcgm_field_meta_t()
    fn = dcgmFP("DcgmFieldGetById")
    fn.restype = POINTER(c_dcgm_field_meta_t)
    c_field_meta_ptr = fn(fieldId)
    if not c_field_meta_ptr:
        return None

    retVal = c_dcgm_field_meta_t()
    memmove(addressof(retVal), c_field_meta_ptr, sizeof(retVal))
    return retVal

def DcgmFieldGetByTag(tag):
    '''
    Get metadata for a field, given its string tag

    :param tag: Field tag to get metadata for. Example 'brand'
    :return: c_dcgm_field_meta_t struct on success. None on error.
    '''
    DcgmFieldsInit()

    retVal = c_dcgm_field_meta_t()
    fn = dcgmFP("DcgmFieldGetByTag")
    fn.restype = POINTER(c_dcgm_field_meta_t)
    c_field_meta_ptr = fn(c_char_p(tag.encode('utf-8')))
    if not c_field_meta_ptr:
        return None

    retVal = c_dcgm_field_meta_t()
    memmove(addressof(retVal), c_field_meta_ptr, sizeof(retVal))
    return retVal

def DcgmFieldGetTagById(fieldId):
    field = DcgmFieldGetById(fieldId)
    if field:
        return field.tag
    else:
        return None

