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

from . import dcgm_fields
from . import dcgm_structs
from . import dcgm_agent
from . import dcgm_value as dcgmvalue
from . import dcgm_fields_internal

import time
import ctypes
import json

# @Yueming Hao: add this warpper class to make is consistent with the latest dcgm_filed_helpers.py


class DcgmFieldGroup:
    def __init__(self, dcgm_handle, field_ids, group_name, fieldGroupId):
        self.dcgm_handle = dcgm_handle
        self.field_ids = field_ids
        self.group_name = group_name
        self.fieldGroupId = fieldGroupId


'''
Helper class that makes a python-friendly field value from one returned from the python bindings
'''
class DcgmFieldValue():
    '''
    Constructor

    rawValue is the latest dcgm_structs.c_dcgmFieldValue_v? structure of a field value returned from the raw APIs
    '''
    def __init__(self, rawValue):
        #Make sure the class passed in is an expected type
        if not type(rawValue) == dcgm_structs.c_dcgmFieldValue_v1:
            raise Exception("Unexpected rawValue type %s" % str(type(rawValue)))

        self.ts = rawValue.ts
        self.fieldId = rawValue.fieldId
        self.fieldType = chr(rawValue.fieldType)
        self.isBlank = False
        self.value = None

        if rawValue.status != dcgm_structs.DCGM_ST_OK:
            self.isBlank = True
            return

        if self.fieldType == dcgm_fields.DCGM_FT_DOUBLE:
            self.value = float(rawValue.value.dbl)
            self.isBlank = dcgmvalue.DCGM_FP64_IS_BLANK(self.value)
        elif self.fieldType == dcgm_fields.DCGM_FT_INT64 or self.fieldType == dcgm_fields.DCGM_FT_TIMESTAMP:
            self.value = int(rawValue.value.i64)
            self.isBlank = dcgmvalue.DCGM_INT64_IS_BLANK(self.value)
        elif self.fieldType == dcgm_fields.DCGM_FT_STRING:
            self.value = str(rawValue.value.str)
            self.isBlank = dcgmvalue.DCGM_STR_IS_BLANK(self.value)
        elif self.fieldType == dcgm_fields.DCGM_FT_BINARY:
            if self.fieldId == dcgm_fields.DCGM_FI_DEV_ACCOUNTING_DATA:
                accStats = dcgm_structs.c_dcgmDevicePidAccountingStats_v1()
                ctypes.memmove(ctypes.addressof(accStats), rawValue.value.blob, accStats.FieldsSizeof())
            if self.fieldId in [dcgm_fields_internal.DCGM_FI_DEV_COMPUTE_PIDS, dcgm_fields_internal.DCGM_FI_DEV_GRAPHICS_PIDS]:
                processStats = dcgm_structs.c_dcgmRunningProcess_t()
                ctypes.memmove(ctypes.addressof(processStats), rawValue.value.blob, processStats.FieldsSizeof())
                self.value = processStats
                self.fieldType = dcgm_fields.DCGM_FT_BINARY
                # This should always be false
                self.isBlank = dcgmvalue.DCGM_INT64_IS_BLANK(processStats.pid)
            elif self.fieldId == dcgm_fields.DCGM_FI_SYNC_BOOST:
                #Not exposed publicly for now
                self.value = None
            else:
                raise Exception("Blobs not handled yet for fieldId %d" % self.fieldId)
        else:
            raise Exception("Unhandled fieldType: %s" % self.fieldType)

class DcgmFieldValueTimeSeries:
    def __init__(self):
        self.values = [] #Values in timestamp order

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def InsertValue(self, value):
        if len(self.values) < 1 or value.ts >= self.values[-1].ts:
            self.values.append(value)
            return

        #Otherwise, we need to insert the value in the correct place. Find the place
        for i, existingValue in enumerate(self.values):
            if value.ts < existingValue.ts:
                self.values.insert(i, value)
                return

        raise Exception("Unexpected no place to insert ts %d" % value.ts)

class FieldValueEncoder(json.JSONEncoder):
    # Pylint does not link overloading the default method, so the comment below is WAR for the linting problem
    def default(self, obj): # pylint: disable=E0202
        nested_json = []
        i=0
        for key in obj:
            if isinstance(key, DcgmFieldValue):
                if(key.isBlank):
                    continue
                nested_json.append({'Timestamp' : key.ts, 'FieldId': key.fieldId, 'Value' : key.value})
            else:
                return json.JSONEncoder.default(self, obj) # Let default encoder throw exception    
        return nested_json


def py_helper_dcgm_field_values_since_callback(gpuId, values, numValues, userData):

    userData = ctypes.cast(userData, ctypes.py_object).value
    userData._ProcessValues(gpuId, values[0:numValues])
    return 0

helper_dcgm_field_values_since_callback = dcgm_agent.dcgmFieldValueEnumeration_f(py_helper_dcgm_field_values_since_callback)

def py_helper_dcgm_field_values_since_callback_v2(entityGroupId, entityId, values, numValues, userData):
    userData = ctypes.cast(userData, ctypes.py_object).value
    userData._ProcessValues(entityGroupId, entityId, values[0:numValues])
    return 0

helper_dcgm_field_values_since_callback_v2 = dcgm_agent.dcgmFieldValueEntityEnumeration_f(py_helper_dcgm_field_values_since_callback_v2)

'''
Helper class for handling field value update callbacks and storing them in a .values member variable
'''
class DcgmFieldValueCollection:
    def __init__(self, handle, groupId):
        self.values = {} #2D dictionary of [gpuId][fieldId](DcgmFieldValueTimeSeries)
        self._handle = handle
        self._groupId = groupId
        self._numValuesSeen = 0
        self._nextSinceTimestamp = 0

    '''
    Helper function called by the callback of dcgm_agent.dcgmGetValuesSince to process individual field values
    '''
    def _ProcessValues(self, gpuId, values):
        self._numValuesSeen += len(values)

        if gpuId not in self.values:
            self.values[gpuId] = {}

        for rawValue in values:
            #Convert to python-friendly value
            value = DcgmFieldValue(rawValue)

            if value.fieldId not in self.values[gpuId]:
                self.values[gpuId][value.fieldId] = DcgmFieldValueTimeSeries()

            self.values[gpuId][value.fieldId].InsertValue(value)

    '''
    Get the latest values for a fieldGroup and store them to the .values member variable

    Note: This class does not automatically watch fieldGroup. You must do that ahead of time with dcgmGroup.samples.WatchFields()
    '''
    def GetLatestValues(self, fieldGroup):
        ret = dcgm_agent.dcgmGetLatestValues(self._handle, self._groupId, fieldGroup.fieldGroupId, helper_dcgm_field_values_since_callback, self)
        #Will throw exception on error
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Method to cause more field values to be retrieved from DCGM. Returns the
    number of field values that were retrieved.
    '''
    def GetAllSinceLastCall(self, fieldGroup):
        beforeCount = self._numValuesSeen
        self._nextSinceTimestamp = dcgm_agent.dcgmGetValuesSince(self._handle, self._groupId, fieldGroup.fieldGroupId, self._nextSinceTimestamp, helper_dcgm_field_values_since_callback, self)
        afterCount = self._numValuesSeen
        return afterCount - beforeCount

    def GetLatestValues_v2(self, fieldGroup):
        ret = dcgm_agent.dcgmGetLatestValues_v2(self._handle, self._groupId, fieldGroup.fieldGroupId, helper_dcgm_field_values_since_callback_v2, self)
        #Will throw exception on error
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Method to cause more field values to be retrieved from DCGM. Returns the number of field values that were retrieved
    '''
    def GetAllSinceLastCall_v2(self, fieldGroup):
        beforeCount = self._numValuesSeen
        self._nextSinceTimestamp = dcgm_agent.dcgmGetValuesSince_v2(self._handle, self._groupId, fieldGroup.fieldGroupId, self._nextSinceTimestamp, helper_dcgm_field_values_since_entity_callback, self)
        afterCount = self._numValuesSeen
        return afterCount - beforeCount
        

    '''
    Empty .values{} so that old data is no longer present in this structure.
    This can be used to prevent .values from growing over time
    '''
    def EmptyValues(self):
        self.values = {}
        self._numValuesSeen = 0


'''
Helper class for watching a field group and storing fields values returned from it
'''
class DcgmFieldGroupWatcher(DcgmFieldValueCollection):
    '''
    Constructor

    handle is a DCGM handle from dcgm_agent.dcgmInit()
    groupId is a valid DCGM group ID returned from dcgm_agent.dcgmGroupCreate
    fieldGroup is the DcgmFieldGroup() instance to watch fields for
    operationMode is a dcgm_structs.DCGM_OPERATION_MODE_? constant for if the host engine is running in lock step or auto mode
    updateFreq is how often to update each field in usec
    maxKeepAge is how long DCGM should keep values for in seconds
    maxKeepSamples is the maximum number of samples DCGM should ever cache for each field
    startTimestamp is a base timestamp we should start from when first reading values. This can be used to resume a
                   previous instance of a DcgmFieldGroupWatcher by using its _nextSinceTimestamp.
                   0=start with all cached data
    '''
    def __init__(self, handle, groupId, fieldGroup, operationMode, updateFreq, maxKeepAge, maxKeepSamples, startTimestamp):
        self._fieldGroup = fieldGroup
        self._operationMode = operationMode
        self._updateFreq = updateFreq
        self._maxKeepAge = maxKeepAge
        self._maxKeepSamples = maxKeepSamples
        DcgmFieldValueCollection.__init__(self, handle, groupId)

        self._nextSinceTimestamp = 0 #Start from beginning of time
        if startTimestamp > 0:
            self._nextSinceTimestamp = startTimestamp
        self._numValuesSeen = 0

        #Start watches
        self._WatchFieldGroup()

    '''
    Initiate the host engine watch on the fields
    '''
    def _WatchFieldGroup(self):
        ret = dcgm_agent.dcgmWatchFields(self._handle, self._groupId, self._fieldGroup.fieldGroupId, self._updateFreq, self._maxKeepAge, self._maxKeepSamples)
        dcgm_structs._dcgmCheckReturn(ret) #Will throw exception on error

        # Force an update of the fields so that we can fetch initial values.
        ret = dcgm_agent.dcgmUpdateAllFields(self._handle, 1)
        dcgm_structs._dcgmCheckReturn(ret) #Will throw exception on error

        # Initial update will fetch from startTimestamp.
        self.GetAllSinceLastCall()

    '''
    Method to cause more field values to be retrieved from DCGM. Returns the
    number of field values that were retrieved
    '''
    def GetAllSinceLastCall(self):
        #If we're in manual mode, force an update
        if self._operationMode == dcgm_structs.DCGM_OPERATION_MODE_MANUAL:
            ret = dcgm_agent.dcgmUpdateAllFields(self._handle, 1)
            dcgm_structs._dcgmCheckReturn(ret) #Will throw exception on error

        return super().GetAllSinceLastCall(self._fieldGroup)

    
def py_helper_dcgm_field_values_since_entity_callback(entityGroupId, entityId, values, numValues, userData):

    userData = ctypes.cast(userData, ctypes.py_object).value
    userData._ProcessValues(entityGroupId, entityId, values[0:numValues])
    return 0

helper_dcgm_field_values_since_entity_callback = dcgm_agent.dcgmFieldValueEntityEnumeration_f(py_helper_dcgm_field_values_since_entity_callback)

'''
Helper class for handling field value update callbacks and storing them in a .values member variable
'''
class DcgmFieldValueEntityCollection:
    def __init__(self, handle, groupId):
        self.values = {} #3D dictionary of [entityGroupId][entityId][fieldId](DcgmFieldValueTimeSeries)
        self._handle = handle
        self._groupId = groupId
        self._numValuesSeen = 0
        self._nextSinceTimestamp = 0
        

    '''
    Helper function called by the callback of dcgm_agent.dcgmGetValuesSince to process individual field values
    '''
    def _ProcessValues(self, entityGroupId, entityId, values):
        self._numValuesSeen += len(values)

        if entityGroupId not in self.values:
            self.values[entityGroupId] = {}

        if entityId not in self.values[entityGroupId]:
            self.values[entityGroupId][entityId] = {}

        for rawValue in values:
            #Convert to python-friendly value
            value = DcgmFieldValue(rawValue)

            if value.fieldId not in self.values[entityGroupId][entityId]:
                self.values[entityGroupId][entityId][value.fieldId] = DcgmFieldValueTimeSeries()

            self.values[entityGroupId][entityId][value.fieldId].InsertValue(value)

    '''
    Get the latest values for a fieldGroup and store them to the .values member variable

    Note: This class does not automatically watch fieldGroup. You must do that ahead of time with dcgmGroup.samples.WatchFields()
    '''
    def GetLatestValues(self, fieldGroup):
        ret = dcgm_agent.dcgmGetLatestValues_v2(self._handle, self._groupId, fieldGroup.fieldGroupId, helper_dcgm_field_values_since_entity_callback, self)
        #Will throw exception on error
        dcgm_structs._dcgmCheckReturn(ret)

    '''
    Method to cause more field values to be retrieved from DCGM. Returns the
    number of field values that were retrieved.
    '''
    def GetAllSinceLastCall(self, fieldGroup):
        beforeCount = self._numValuesSeen
        self._nextSinceTimestamp = dcgm_agent.dcgmGetValuesSince_v2(self._handle, self._groupId, fieldGroup.fieldGroupId, self._nextSinceTimestamp, helper_dcgm_field_values_since_entity_callback, self)
        afterCount = self._numValuesSeen
        return afterCount - beforeCount
        
    
    '''
    Empty .values{} so that old data is no longer present in this structure.
    This can be used to prevent .values from growing over time
    '''
    def EmptyValues(self):
        self.values = {}
        self._numValuesSeen = 0


'''
Helper class for watching a field group and storing fields values returned from it
'''
class DcgmFieldGroupEntityWatcher(DcgmFieldValueEntityCollection):
    '''
    Constructor

    handle is a DCGM handle from dcgm_agent.dcgmInit()
    groupId is a valid DCGM group ID returned from dcgm_agent.dcgmGroupCreate
    fieldGroup is the DcgmFieldGroup() instance to watch fields for
    operationMode is a dcgm_structs.DCGM_OPERATION_MODE_? constant for if the host engine is running in lock step or auto mode
    updateFreq is how often to update each field in usec
    maxKeepAge is how long DCGM should keep values for in seconds
    maxKeepSamples is the maximum number of samples DCGM should ever cache for each field
    startTimestamp is a base timestamp we should start from when first reading values. This can be used to resume a
                   previous instance of a DcgmFieldGroupWatcher by using its _nextSinceTimestamp.
                   0=start with all cached data
    '''
    def __init__(self, handle, groupId, fieldGroup, operationMode, updateFreq, maxKeepAge, maxKeepSamples, startTimestamp):
        self._fieldGroup = fieldGroup
        self._operationMode = operationMode
        self._updateFreq = updateFreq
        self._maxKeepAge = maxKeepAge
        self._maxKeepSamples = maxKeepSamples
        DcgmFieldValueEntityCollection.__init__(self, handle, groupId)

        self._nextSinceTimestamp = 0 #Start from beginning of time
        if startTimestamp > 0:
            self._nextSinceTimestamp = startTimestamp
        self._numValuesSeen = 0

        #Start watches
        self._WatchFieldGroup()

    '''
    Initiate the host engine watch on the fields
    '''
    def _WatchFieldGroup(self):
        ret = dcgm_agent.dcgmWatchFields(self._handle, self._groupId, self._fieldGroup.fieldGroupId, self._updateFreq, self._maxKeepAge, self._maxKeepSamples)
        dcgm_structs._dcgmCheckReturn(ret) #Will throw exception on error

        # Force an update of the fields so that we can fetch initial values.
        ret = dcgm_agent.dcgmUpdateAllFields(self._handle, 1)
        dcgm_structs._dcgmCheckReturn(ret) #Will throw exception on error

        # Initial update will fetch from startTimestamp.
        self.GetAllSinceLastCall()

    '''
    Method to cause more field values to be retrieved from DCGM. Returns the
    number of field values that were retrieved
    '''
    def GetAllSinceLastCall(self):
        #If we're in manual mode, force an update
        if self._operationMode == dcgm_structs.DCGM_OPERATION_MODE_MANUAL:
            ret = dcgm_agent.dcgmUpdateAllFields(self._handle, 1)
            dcgm_structs._dcgmCheckReturn(ret) #Will throw exception on error

        return super().GetAllSinceLastCall(self._fieldGroup)

#Test program for demonstrating how this module works
# def main():
#     operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
#     timeStep = 1.0

#     dcgm_structs._dcgmInit()
#     dcgm_agent.dcgmInit() #Will throw an exception on error
#     handle = dcgm_agent.dcgmStartEmbedded(operationMode)
#     handleObj = pydcgm.DcgmHandle(handle=handle)
#     groupId = dcgm_structs.DCGM_GROUP_ALL_GPUS
#     fieldIds = [dcgm_fields.DCGM_FI_DEV_SM_CLOCK, dcgm_fields.DCGM_FI_DEV_MEM_CLOCK]

#     fieldGroup = pydcgm.DcgmFieldGroup(handleObj, "my_field_group", fieldIds)

#     updateFreq = int(timeStep * 1000000.0)
#     maxKeepAge = 3600.0 #1 hour
#     maxKeepSamples = 0 #unlimited. maxKeepAge will enforce quota
#     startTimestamp = 0 #beginning of time

#     dfcw = DcgmFieldGroupWatcher(handle, groupId, fieldGroup, operationMode, updateFreq, maxKeepAge, maxKeepSamples, startTimestamp)
#     dfcw2 = DcgmFieldGroupEntityWatcher(handle, groupId, fieldGroup, operationMode, updateFreq, maxKeepAge, maxKeepSamples, startTimestamp)

#     while(True):
#         newUpdateCount = dfcw.GetAllSinceLastCall()
#         newUpdateCount2 = dfcw2.GetAllSinceLastCall()
#         print("Got %d and %d new field value updates" % (newUpdateCount, newUpdateCount2))
#         for gpuId in list(dfcw.values.keys()):
#             print("gpuId %d" % gpuId)
#             for fieldId in list(dfcw.values[gpuId].keys()):
#                 print("    fieldId %d: %d values. latest timestamp %d" % \
#                       (fieldId, len(dfcw.values[gpuId][fieldId]), dfcw.values[gpuId][fieldId][-1].ts))

#         for entityGroupId in list(dfcw2.values.keys()):
#             print("entityGroupId %d" % entityGroupId)
#             for entityId in list(dfcw2.values[entityGroupId].keys()):
#                 print("    entityId %d" % entityId)
#                 for fieldId in list(dfcw2.values[entityGroupId][entityId].keys()):
#                     print("        fieldId %d: %d values. latest timestamp %d" % \
#                           (fieldId, len(dfcw2.values[entityGroupId][entityId][fieldId]), dfcw2.values[entityGroupId][entityId][fieldId][-1].ts))

#         time.sleep(timeStep)

# if __name__ == "__main__":
#     main()
