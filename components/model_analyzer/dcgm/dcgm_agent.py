# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from . import dcgm_structs
from ctypes import *
# Provides access to functions from dcgm_agent_internal
dcgmFP = dcgm_structs._dcgmGetFunctionPointer


# This method is used to initialize DCGM
def dcgmInit():
    dcgm_handle = c_void_p()
    fn = dcgmFP("dcgmInit")
    ret = fn(byref(dcgm_handle))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


# This method is used to shutdown DCGM Engine
def dcgmShutdown():
    fn = dcgmFP("dcgmShutdown")
    ret = fn()
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmStartEmbedded(opMode):
    dcgm_handle = c_void_p()
    fn = dcgmFP("dcgmStartEmbedded")
    ret = fn(opMode, byref(dcgm_handle))
    dcgm_structs._dcgmCheckReturn(ret)
    return dcgm_handle


def dcgmStopEmbedded(dcgm_handle):
    fn = dcgmFP("dcgmStopEmbedded")
    ret = fn(dcgm_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmConnect(ip_address):
    dcgm_handle = c_void_p()
    fn = dcgmFP("dcgmConnect")
    ret = fn(ip_address, byref(dcgm_handle))
    dcgm_structs._dcgmCheckReturn(ret)
    return dcgm_handle


def dcgmConnect_v2(ip_address,
                   connectParams,
                   version=dcgm_structs.c_dcgmConnectV2Params_version):
    connectParams.version = version
    dcgm_handle = c_void_p()
    fn = dcgmFP("dcgmConnect_v2")
    ret = fn(ip_address, byref(connectParams), byref(dcgm_handle))
    dcgm_structs._dcgmCheckReturn(ret)
    return dcgm_handle


def dcgmDisconnect(dcgm_handle):
    fn = dcgmFP("dcgmDisconnect")
    ret = fn(dcgm_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGetAllSupportedDevices(dcgm_handle):
    c_count = c_uint()
    gpuid_list = c_uint * dcgm_structs.DCGM_MAX_NUM_DEVICES
    c_gpuid_list = gpuid_list()
    fn = dcgmFP("dcgmGetAllSupportedDevices")
    ret = fn(dcgm_handle, c_gpuid_list, byref(c_count))
    dcgm_structs._dcgmCheckReturn(ret)
    return [c_gpuid_list[i] for i in range(c_count.value)[0:int(c_count.value)]]


def dcgmGetAllDevices(dcgm_handle):
    c_count = c_uint()
    gpuid_list = c_uint * dcgm_structs.DCGM_MAX_NUM_DEVICES
    c_gpuid_list = gpuid_list()
    fn = dcgmFP("dcgmGetAllDevices")
    ret = fn(dcgm_handle, c_gpuid_list, byref(c_count))
    dcgm_structs._dcgmCheckReturn(ret)
    return [c_gpuid_list[i] for i in range(c_count.value)[0:int(c_count.value)]]


def dcgmGetDeviceAttributes(dcgm_handle, gpuId):
    fn = dcgmFP("dcgmGetDeviceAttributes")
    device_values = dcgm_structs.c_dcgmDeviceAttributes_v2()
    device_values.version = dcgm_structs.dcgmDeviceAttributes_version2
    ret = fn(dcgm_handle, c_int(gpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values


def dcgmGetEntityGroupEntities(dcgm_handle, entityGroup, flags):
    capacity = dcgm_structs.DCGM_GROUP_MAX_ENTITIES
    c_count = c_int32(capacity)
    entityIds = c_uint32 * capacity
    c_entityIds = entityIds()
    fn = dcgmFP("dcgmGetEntityGroupEntities")
    ret = fn(dcgm_handle, entityGroup, c_entityIds, byref(c_count), flags)
    dcgm_structs._dcgmCheckReturn(ret)
    return c_entityIds[0:int(c_count.value)]


def dcgmGetNvLinkLinkStatus(dcgm_handle):
    linkStatus = dcgm_structs.c_dcgmNvLinkStatus_v2()
    linkStatus.version = dcgm_structs.dcgmNvLinkStatus_version2
    fn = dcgmFP("dcgmGetNvLinkLinkStatus")
    ret = fn(dcgm_handle, byref(linkStatus))
    dcgm_structs._dcgmCheckReturn(ret)
    return linkStatus


def dcgmGetGpuInstanceHierarchy(dcgm_handle):
    hierarchy = dcgm_structs.c_dcgmMigHierarchy_v1()
    hierarchy.version = dcgm_structs.c_dcgmMigHierarchy_version1
    fn = dcgmFP("dcgmGetGpuInstanceHierarchy")
    ret = fn(dcgm_handle, byref(hierarchy))
    dcgm_structs._dcgmCheckReturn(ret)
    return hierarchy


def dcgmCreateMigEntity(dcgm_handle, parentId, profile, createOption, flags):
    fn = dcgmFP("dcgmCreateMigEntity")
    cme = dcgm_structs.c_dcgmCreateMigEntity_v1()
    cme.version = dcgm_structs.c_dcgmCreateMigEntity_version1
    cme.parentId = parentId
    cme.createOption = createOption
    cme.profile = profile
    cme.flags = flags
    ret = fn(dcgm_handle, byref(cme))
    dcgm_structs._dcgmCheckReturn(ret)


def dcgmDeleteMigEntity(dcgm_handle, entityGroupId, entityId, flags):
    fn = dcgmFP("dcgmDeleteMigEntity")
    dme = dcgm_structs.c_dcgmDeleteMigEntity_v1()
    dme.version = dcgm_structs.c_dcgmDeleteMigEntity_version1
    dme.entityGroupId = entityGroupId
    dme.entityId = entityId
    dme.flags = flags
    ret = fn(dcgm_handle, byref(dme))
    dcgm_structs._dcgmCheckReturn(ret)


def dcgmGroupCreate(dcgm_handle, type, groupName):
    c_group_id = c_void_p()
    fn = dcgmFP("dcgmGroupCreate")
    ret = fn(dcgm_handle, type, groupName, byref(c_group_id))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_group_id


def dcgmGroupDestroy(dcgm_handle, group_id):
    fn = dcgmFP("dcgmGroupDestroy")
    ret = fn(dcgm_handle, group_id)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGroupAddDevice(dcgm_handle, group_id, gpu_id):
    fn = dcgmFP("dcgmGroupAddDevice")
    ret = fn(dcgm_handle, group_id, gpu_id)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGroupAddEntity(dcgm_handle, group_id, entityGroupId, entityId):
    fn = dcgmFP("dcgmGroupAddEntity")
    ret = fn(dcgm_handle, group_id, entityGroupId, entityId)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGroupRemoveDevice(dcgm_handle, group_id, gpu_id):
    fn = dcgmFP("dcgmGroupRemoveDevice")
    ret = fn(dcgm_handle, group_id, gpu_id)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGroupRemoveEntity(dcgm_handle, group_id, entityGroupId, entityId):
    fn = dcgmFP("dcgmGroupRemoveEntity")
    ret = fn(dcgm_handle, group_id, entityGroupId, entityId)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGroupGetInfo(dcgm_handle,
                     group_id,
                     version=dcgm_structs.c_dcgmGroupInfo_version2):
    fn = dcgmFP("dcgmGroupGetInfo")

    # support the old version of the request since the host engine does
    if version == dcgm_structs.c_dcgmGroupInfo_version2:
        device_values = dcgm_structs.c_dcgmGroupInfo_v2()
        device_values.version = dcgm_structs.c_dcgmGroupInfo_version2
    else:
        dcgm_structs._dcgmCheckReturn(dcgm_structs.DCGM_ST_VER_MISMATCH)

    ret = fn(dcgm_handle, group_id, byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values


def dcgmGroupGetAllIds(dcgmHandle):
    fn = dcgmFP("dcgmGroupGetAllIds")
    c_count = c_uint()
    groupIdList = c_void_p * dcgm_structs.DCGM_MAX_NUM_GROUPS
    c_groupIdList = groupIdList()
    ret = fn(dcgmHandle, c_groupIdList, byref(c_count))
    dcgm_structs._dcgmCheckReturn(ret)
    return map(None, c_groupIdList[0:int(c_count.value)])


def dcgmFieldGroupCreate(dcgm_handle, fieldIds, fieldGroupName):
    c_field_group_id = c_void_p()
    c_num_field_ids = c_int32(len(fieldIds))
    c_field_ids = (c_uint16 * len(fieldIds))(*fieldIds)
    fn = dcgmFP("dcgmFieldGroupCreate")
    ret = fn(dcgm_handle, c_num_field_ids, byref(c_field_ids), fieldGroupName,
             byref(c_field_group_id))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_field_group_id


def dcgmFieldGroupDestroy(dcgm_handle, fieldGroupId):
    fn = dcgmFP("dcgmFieldGroupDestroy")
    ret = fn(dcgm_handle, fieldGroupId)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmFieldGroupGetInfo(dcgm_handle, fieldGroupId):
    c_fieldGroupInfo = dcgm_structs.c_dcgmFieldGroupInfo_v1()
    c_fieldGroupInfo.version = dcgm_structs.dcgmFieldGroupInfo_version1
    c_fieldGroupInfo.fieldGroupId = fieldGroupId
    fn = dcgmFP("dcgmFieldGroupGetInfo")
    ret = fn(dcgm_handle, byref(c_fieldGroupInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_fieldGroupInfo


def dcgmFieldGroupGetAll(dcgm_handle):
    c_allGroupInfo = dcgm_structs.c_dcgmAllFieldGroup_v1()
    c_allGroupInfo.version = dcgm_structs.dcgmAllFieldGroup_version1
    fn = dcgmFP("dcgmFieldGroupGetAll")
    ret = fn(dcgm_handle, byref(c_allGroupInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_allGroupInfo


def dcgmStatusCreate():
    c_status_handle = c_void_p()
    fn = dcgmFP("dcgmStatusCreate")
    ret = fn(byref(c_status_handle))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_status_handle


def dcgmStatusDestroy(status_handle):
    fn = dcgmFP("dcgmStatusDestroy")
    ret = fn(status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmStatusGetCount(status_handle):
    c_count = c_uint()
    fn = dcgmFP("dcgmStatusGetCount")
    ret = fn(status_handle, byref(c_count))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_count.value


def dcgmStatusPopError(status_handle):
    c_errorInfo = dcgm_structs.c_dcgmErrorInfo_v1()
    fn = dcgmFP("dcgmStatusPopError")
    ret = fn(status_handle, byref(c_errorInfo))
    if ret == dcgm_structs.DCGM_ST_OK:
        return c_errorInfo
    else:
        return None
    return c_errorInfo


def dcgmStatusClear(status_handle):
    fn = dcgmFP("dcgmStatusClear")
    ret = fn(status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmConfigSet(dcgm_handle, group_id, configToSet, status_handle):
    fn = dcgmFP("dcgmConfigSet")
    configToSet.version = dcgm_structs.dcgmDeviceConfig_version1
    ret = fn(dcgm_handle, group_id, byref(configToSet), status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmConfigGet(dcgm_handle, group_id, reqCfgType, count, status_handle):
    fn = dcgmFP("dcgmConfigGet")

    config_values_array = count * dcgm_structs.c_dcgmDeviceConfig_v1
    c_config_values = config_values_array()

    for index in range(0, count):
        c_config_values[index].version = dcgm_structs.dcgmDeviceConfig_version1

    ret = fn(dcgm_handle, group_id, reqCfgType, count, c_config_values,
             status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return map(None, c_config_values[0:count])


def dcgmConfigEnforce(dcgm_handle, group_id, status_handle):
    fn = dcgmFP("dcgmConfigEnforce")
    ret = fn(dcgm_handle, group_id, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


# This method is used to tell the cache manager to update all fields
def dcgmUpdateAllFields(dcgm_handle, waitForUpdate):
    fn = dcgmFP("dcgmUpdateAllFields")
    ret = fn(dcgm_handle, c_int(waitForUpdate))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


# This method is used to get the policy information
def dcgmPolicyGet(dcgm_handle, group_id, count, status_handle):
    fn = dcgmFP("dcgmPolicyGet")
    policy_array = count * dcgm_structs.c_dcgmPolicy_v1

    c_policy_values = policy_array()

    for index in range(0, count):
        c_policy_values[index].version = dcgm_structs.dcgmPolicy_version1

    ret = fn(dcgm_handle, group_id, count, c_policy_values, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return c_policy_values[0:count]


# This method is used to set the policy information
def dcgmPolicySet(dcgm_handle, group_id, policy, status_handle):
    fn = dcgmFP("dcgmPolicySet")
    ret = fn(dcgm_handle, group_id, byref(policy), status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


# First parameter below is the return type
dcgmFieldValueEnumeration_f = CFUNCTYPE(
    c_int32, c_uint32, POINTER(dcgm_structs.c_dcgmFieldValue_v1), c_int32,
    c_void_p)
dcgmFieldValueEntityEnumeration_f = CFUNCTYPE(
    c_int32, c_uint32, c_uint32, POINTER(dcgm_structs.c_dcgmFieldValue_v1),
    c_int32, c_void_p)


def dcgmGetValuesSince(dcgm_handle, groupId, fieldGroupId, sinceTimestamp,
                       enumCB, userData):
    fn = dcgmFP("dcgmGetValuesSince")
    c_nextSinceTimestamp = c_int64()
    ret = fn(dcgm_handle, groupId, fieldGroupId, c_int64(sinceTimestamp),
             byref(c_nextSinceTimestamp), enumCB, py_object(userData))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_nextSinceTimestamp.value


def dcgmGetValuesSince_v2(dcgm_handle, groupId, fieldGroupId, sinceTimestamp,
                          enumCB, userData):
    fn = dcgmFP("dcgmGetValuesSince_v2")
    c_nextSinceTimestamp = c_int64()
    ret = fn(dcgm_handle, groupId, fieldGroupId, c_int64(sinceTimestamp),
             byref(c_nextSinceTimestamp), enumCB, py_object(userData))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_nextSinceTimestamp.value


def dcgmGetLatestValues(dcgm_handle, groupId, fieldGroupId, enumCB, userData):
    fn = dcgmFP("dcgmGetLatestValues")
    ret = fn(dcgm_handle, groupId, fieldGroupId, enumCB, py_object(userData))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGetLatestValues_v2(dcgm_handle, groupId, fieldGroupId, enumCB,
                           userData):
    fn = dcgmFP("dcgmGetLatestValues_v2")
    ret = fn(dcgm_handle, groupId, fieldGroupId, enumCB, py_object(userData))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmWatchFields(dcgm_handle, groupId, fieldGroupId, updateFreq, maxKeepAge,
                    maxKeepSamples):
    fn = dcgmFP("dcgmWatchFields")
    ret = fn(dcgm_handle, groupId, fieldGroupId, c_int64(updateFreq),
             c_double(maxKeepAge), c_int32(maxKeepSamples))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmUnwatchFields(dcgm_handle, groupId, fieldGroupId):
    fn = dcgmFP("dcgmUnwatchFields")
    ret = fn(dcgm_handle, groupId, fieldGroupId)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmHealthSet(dcgm_handle, groupId, systems):
    fn = dcgmFP("dcgmHealthSet")
    ret = fn(dcgm_handle, groupId, systems)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmHealthSet_v2(dcgm_handle, groupId, systems, updateInterval, maxKeepAge):
    params = dcgm_structs.c_dcgmHealthSetParams_v2()
    params.version = dcgm_structs.dcgmHealthSetParams_version2
    params.groupId = groupId
    params.systems = systems
    params.updateInterval = updateInterval
    params.maxKeepAge = maxKeepAge

    fn = dcgmFP("dcgmHealthSet_v2")
    ret = fn(dcgm_handle, byref(params))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmHealthGet(dcgm_handle, groupId):
    c_systems = c_int32()
    fn = dcgmFP("dcgmHealthGet")
    ret = fn(dcgm_handle, groupId, byref(c_systems))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_systems.value


def dcgmHealthCheck(dcgm_handle,
                    groupId,
                    version=dcgm_structs.dcgmHealthResponse_version4):
    if version != dcgm_structs.dcgmHealthResponse_version4:
        dcgm_structs._dcgmCheckReturn(dcgm_structs.DCGM_ST_VER_MISMATCH)

    c_results = dcgm_structs.c_dcgmHealthResponse_v4()
    c_results.version = dcgm_structs.dcgmHealthResponse_version4
    fn = dcgmFP("dcgmHealthCheck")
    ret = fn(dcgm_handle, groupId, byref(c_results))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_results


def dcgmPolicyRegister(dcgm_handle, groupId, condition, beginCallback,
                       finishCallback):
    fn = dcgmFP("dcgmPolicyRegister")
    ret = fn(dcgm_handle, groupId, condition, beginCallback, finishCallback)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmPolicyUnregister(dcgm_handle, groupId, condition):
    fn = dcgmFP("dcgmPolicyUnregister")
    ret = fn(dcgm_handle, groupId, condition)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmPolicyTrigger(dcgm_handle):
    fn = dcgmFP("dcgmPolicyTrigger")
    ret = fn(dcgm_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def helperDiagCheckReturn(ret, response):
    try:
        dcgm_structs._dcgmCheckReturn(ret)
    except dcgm_structs.DCGMError as e:
        if response.systemError.msg != "":
            # Add systemError information to the raised exception.
            import sys
            info = "%s" % response.systemError.msg
            e.SetAdditionalInfo(info)
            raise e  # pylint: disable=E0710
        else:
            raise

    return response


def dcgmActionValidate_v2(dcgm_handle,
                          runDiagInfo,
                          runDiagVersion=dcgm_structs.dcgmRunDiag_version6):
    response = dcgm_structs.c_dcgmDiagResponse_v6()
    runDiagInfo.version = runDiagVersion
    response.version = dcgm_structs.dcgmDiagResponse_version6
    fn = dcgmFP("dcgmActionValidate_v2")
    ret = fn(dcgm_handle, byref(runDiagInfo), byref(response))

    return helperDiagCheckReturn(ret, response)


def dcgmActionValidate(dcgm_handle, group_id, validate):
    response = dcgm_structs.c_dcgmDiagResponse_v6()
    response.version = dcgm_structs.dcgmDiagResponse_version6

    # Put the group_id and validate into a dcgmRunDiag struct
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v6()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version6
    runDiagInfo.validate = validate
    runDiagInfo.groupId = group_id

    fn = dcgmFP("dcgmActionValidate_v2")
    ret = fn(dcgm_handle, byref(runDiagInfo), byref(response))

    return helperDiagCheckReturn(ret, response)


def dcgmRunDiagnostic(dcgm_handle, group_id, diagLevel):
    response = dcgm_structs.c_dcgmDiagResponse_v6()
    response.version = dcgm_structs.dcgmDiagResponse_version6
    fn = dcgmFP("dcgmRunDiagnostic")
    ret = fn(dcgm_handle, group_id, diagLevel, byref(response))

    return helperDiagCheckReturn(ret, response)


def dcgmWatchPidFields(dcgm_handle, groupId, updateFreq, maxKeepAge,
                       maxKeepSamples):
    fn = dcgmFP("dcgmWatchPidFields")
    ret = fn(dcgm_handle, groupId, c_int64(updateFreq), c_double(maxKeepAge),
             c_int32(maxKeepSamples))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmGetPidInfo(dcgm_handle, groupId, pid):
    fn = dcgmFP("dcgmGetPidInfo")
    pidInfo = dcgm_structs.c_dcgmPidInfo_v2()

    pidInfo.version = dcgm_structs.dcgmPidInfo_version2
    pidInfo.pid = pid

    ret = fn(dcgm_handle, groupId, byref(pidInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return pidInfo


def dcgmGetDeviceTopology(dcgm_handle, gpuId):
    devtopo = dcgm_structs.c_dcgmDeviceTopology_v1()
    fn = dcgmFP("dcgmGetDeviceTopology")
    ret = fn(dcgm_handle, gpuId, byref(devtopo))
    dcgm_structs._dcgmCheckReturn(ret)
    return devtopo


def dcgmGetGroupTopology(dcgm_handle, groupId):
    grouptopo = dcgm_structs.c_dcgmGroupTopology_v1()
    fn = dcgmFP("dcgmGetGroupTopology")
    ret = fn(dcgm_handle, groupId, byref(grouptopo))
    dcgm_structs._dcgmCheckReturn(ret)
    return grouptopo


def dcgmWatchJobFields(dcgm_handle, groupId, updateFreq, maxKeepAge,
                       maxKeepSamples):
    fn = dcgmFP("dcgmWatchJobFields")
    ret = fn(dcgm_handle, groupId, c_int64(updateFreq), c_double(maxKeepAge),
             c_int32(maxKeepSamples))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmJobStartStats(dcgm_handle, groupId, jobid):
    fn = dcgmFP("dcgmJobStartStats")
    ret = fn(dcgm_handle, groupId, jobid)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmJobStopStats(dcgm_handle, jobid):
    fn = dcgmFP("dcgmJobStopStats")
    ret = fn(dcgm_handle, jobid)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmJobGetStats(dcgm_handle, jobid):
    fn = dcgmFP("dcgmJobGetStats")
    jobInfo = dcgm_structs.c_dcgmJobInfo_v3()

    jobInfo.version = dcgm_structs.dcgmJobInfo_version3

    ret = fn(dcgm_handle, jobid, byref(jobInfo))
    dcgm_structs._dcgmCheckReturn(ret)
    return jobInfo


def dcgmJobRemove(dcgm_handle, jobid):
    fn = dcgmFP("dcgmJobRemove")
    ret = fn(dcgm_handle, jobid)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmJobRemoveAll(dcgm_handle):
    fn = dcgmFP("dcgmJobRemoveAll")
    ret = fn(dcgm_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmIntrospectToggleState(dcgm_handle, enabledState):
    fn = dcgmFP("dcgmIntrospectToggleState")
    ret = fn(dcgm_handle, enabledState)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmIntrospectGetHostengineMemoryUsage(dcgm_handle, waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetHostengineMemoryUsage")

    memInfo = dcgm_structs.c_dcgmIntrospectMemory_v1()
    memInfo.version = dcgm_structs.dcgmIntrospectMemory_version1

    ret = fn(dcgm_handle, byref(memInfo), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo


def dcgmIntrospectGetHostengineCpuUtilization(dcgm_handle, waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetHostengineCpuUtilization")

    cpuUtil = dcgm_structs.c_dcgmIntrospectCpuUtil_v1()
    cpuUtil.version = dcgm_structs.dcgmIntrospectCpuUtil_version1

    ret = fn(dcgm_handle, byref(cpuUtil), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return cpuUtil


def dcgmIntrospectGetFieldsExecTime(dcgm_handle,
                                    introspectContext,
                                    waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetFieldsExecTime")

    execTime = dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v2()
    execTime.version = dcgm_structs.dcgmIntrospectFullFieldsExecTime_version2

    ret = fn(dcgm_handle, byref(introspectContext), byref(execTime),
             waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return execTime


def dcgmIntrospectGetFieldsMemoryUsage(dcgm_handle,
                                       introspectContext,
                                       waitIfNoData=True):
    fn = dcgmFP("dcgmIntrospectGetFieldsMemoryUsage")

    memInfo = dcgm_structs.c_dcgmIntrospectFullMemory_v1()
    memInfo.version = dcgm_structs.dcgmIntrospectFullMemory_version1

    ret = fn(dcgm_handle, byref(introspectContext), byref(memInfo),
             waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo


def dcgmIntrospectUpdateAll(dcgmHandle, waitForUpdate):
    fn = dcgmFP("dcgmIntrospectUpdateAll")
    ret = fn(dcgmHandle, c_int(waitForUpdate))
    dcgm_structs._dcgmCheckReturn(ret)


def dcgmEntityGetLatestValues(dcgmHandle, entityGroup, entityId, fieldIds):
    fn = dcgmFP("dcgmEntityGetLatestValues")
    field_values = (dcgm_structs.c_dcgmFieldValue_v1 * len(fieldIds))()
    id_values = (c_uint16 * len(fieldIds))(*fieldIds)
    ret = fn(dcgmHandle, c_uint(entityGroup),
             dcgm_fields.c_dcgm_field_eid_t(entityId), id_values,
             c_uint(len(fieldIds)), field_values)
    dcgm_structs._dcgmCheckReturn(ret)
    return field_values


def dcgmEntitiesGetLatestValues(dcgmHandle, entities, fieldIds, flags):
    fn = dcgmFP("dcgmEntitiesGetLatestValues")
    numFvs = len(fieldIds) * len(entities)
    field_values = (dcgm_structs.c_dcgmFieldValue_v2 * numFvs)()
    entities_values = (dcgm_structs.c_dcgmGroupEntityPair_t *
                       len(entities))(*entities)
    field_id_values = (c_uint16 * len(fieldIds))(*fieldIds)
    ret = fn(dcgmHandle, entities_values, c_uint(len(entities)),
             field_id_values, c_uint(len(fieldIds)), flags, field_values)
    dcgm_structs._dcgmCheckReturn(ret)
    return field_values


def dcgmSelectGpusByTopology(dcgmHandle, inputGpuIds, numGpus, hintFlags):
    fn = dcgmFP("dcgmSelectGpusByTopology")
    outputGpuIds = c_int64()
    ret = fn(dcgmHandle, c_uint64(inputGpuIds), c_uint32(numGpus),
             byref(outputGpuIds), c_uint64(hintFlags))
    dcgm_structs._dcgmCheckReturn(ret)
    return outputGpuIds


def dcgmGetFieldSummary(dcgmHandle, fieldId, entityGroupType, entityId,
                        summaryMask, startTime, endTime):
    fn = dcgmFP("dcgmGetFieldSummary")
    request = dcgm_structs.c_dcgmFieldSummaryRequest_v1()
    request.version = dcgm_structs.dcgmFieldSummaryRequest_version1
    request.fieldId = fieldId
    request.entityGroupType = entityGroupType
    request.entityId = entityId
    request.summaryTypeMask = summaryMask
    request.startTime = startTime
    request.endTime = endTime
    ret = fn(dcgmHandle, byref(request))
    dcgm_structs._dcgmCheckReturn(ret)
    return request


def dcgmModuleBlacklist(dcgmHandle, moduleId):
    fn = dcgmFP("dcgmModuleBlacklist")
    ret = fn(dcgmHandle, c_uint32(moduleId))
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmModuleGetStatuses(dcgmHandle):
    moduleStatuses = dcgm_structs.c_dcgmModuleGetStatuses_v1()
    moduleStatuses.version = dcgm_structs.dcgmModuleGetStatuses_version1
    fn = dcgmFP("dcgmModuleGetStatuses")
    ret = fn(dcgmHandle, byref(moduleStatuses))
    dcgm_structs._dcgmCheckReturn(ret)
    return moduleStatuses


def dcgmProfGetSupportedMetricGroups(dcgmHandle, groupId):
    msg = dcgm_structs.c_dcgmProfGetMetricGroups_v2()
    msg.version = dcgm_structs.dcgmProfGetMetricGroups_version1
    msg.groupId = groupId
    fn = dcgmFP("dcgmProfGetSupportedMetricGroups")
    ret = fn(dcgmHandle, byref(msg))
    dcgm_structs._dcgmCheckReturn(ret)
    return msg


def dcgmProfWatchFields(dcgmHandle, fieldIds, groupId, updateFreq, maxKeepAge,
                        maxKeepSamples):
    msg = dcgm_structs.c_dcgmProfWatchFields_v1()
    msg.version = dcgm_structs.dcgmProfWatchFields_version1
    msg.groupId = groupId
    msg.updateFreq = updateFreq
    msg.maxKeepAge = maxKeepAge
    msg.maxKeepSamples = maxKeepSamples
    msg.numFieldIds = c_uint32(len(fieldIds))
    for i, fieldId in enumerate(fieldIds):
        msg.fieldIds[i] = fieldId

    fn = dcgmFP("dcgmProfWatchFields")
    ret = fn(dcgmHandle, byref(msg))
    dcgm_structs._dcgmCheckReturn(ret)
    return msg


def dcgmProfUnwatchFields(dcgmHandle, groupId):
    msg = dcgm_structs.c_dcgmProfUnwatchFields_v1()
    msg.version = dcgm_structs.dcgmProfUnwatchFields_version1
    msg.groupId = groupId
    fn = dcgmFP("dcgmProfUnwatchFields")
    ret = fn(dcgmHandle, byref(msg))
    dcgm_structs._dcgmCheckReturn(ret)
    return msg


def dcgmProfPause(dcgmHandle):
    fn = dcgmFP("dcgmProfPause")
    ret = fn(dcgmHandle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmProfResume(dcgmHandle):
    fn = dcgmFP("dcgmProfResume")
    ret = fn(dcgmHandle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret


def dcgmVersionInfo():
    msg = dcgm_structs.c_dcgmVersionInfo_v2()
    msg.version = dcgm_structs.dcgmVersionInfo_version2
    fn = dcgmFP("dcgmVersionInfo")
    ret = fn(byref(msg))
    dcgm_structs._dcgmCheckReturn(ret)
    return msg


def dcgmHostengineIsHealthy(dcgmHandle):
    heHealth = dcgm_structs.c_dcgmHostengineHealth_v1()
    heHealth.version = dcgm_structs.dcgmHostengineHealth_version1
    fn = dcgmFP("dcgmHostengineIsHealthy")
    ret = fn(dcgmHandle, byref(heHealth))
    dcgm_structs._dcgmCheckReturn(ret)
    return heHealth
