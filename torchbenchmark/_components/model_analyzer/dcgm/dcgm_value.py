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

# Base value for integer blank. can be used as an unspecified blank
DCGM_INT32_BLANK = 0x7ffffff0
DCGM_INT64_BLANK = 0x7ffffffffffffff0

# Base value for double blank. 2 ** 47. FP 64 has 52 bits of mantissa,
#so 47 bits can still increment by 1 and represent each value from 0-15
DCGM_FP64_BLANK = 140737488355328.0

DCGM_STR_BLANK = "<<<NULL>>>"

# Represents an error where data was not found
DCGM_INT32_NOT_FOUND = (DCGM_INT32_BLANK+1)
DCGM_INT64_NOT_FOUND = (DCGM_INT64_BLANK+1)
DCGM_FP64_NOT_FOUND = (DCGM_FP64_BLANK+1.0)
DCGM_STR_NOT_FOUND = "<<<NOT_FOUND>>>"

# Represents an error where fetching the value is not supported
DCGM_INT32_NOT_SUPPORTED = (DCGM_INT32_BLANK+2)
DCGM_INT64_NOT_SUPPORTED = (DCGM_INT64_BLANK+2)
DCGM_FP64_NOT_SUPPORTED = (DCGM_FP64_BLANK+2.0)
DCGM_STR_NOT_SUPPORTED = "<<<NOT_SUPPORTED>>>"

# Represents and error where fetching the value is not allowed with our current credentials
DCGM_INT32_NOT_PERMISSIONED = (DCGM_INT32_BLANK+3)
DCGM_INT64_NOT_PERMISSIONED = (DCGM_INT64_BLANK+3)
DCGM_FP64_NOT_PERMISSIONED = (DCGM_FP64_BLANK+3.0)
DCGM_STR_NOT_PERMISSIONED = "<<<NOT_PERM>>>"

###############################################################################
# Functions to check if a value is blank or not
def DCGM_INT32_IS_BLANK(val): 
    if val >= DCGM_INT32_BLANK:
        return True
    else:
        return False

def DCGM_INT64_IS_BLANK(val):
    if val >= DCGM_INT64_BLANK:
        return True
    else:
        return False

def DCGM_FP64_IS_BLANK(val):
    if val >= DCGM_FP64_BLANK:
        return True
    else:
        return False

#Looks for <<< at first position and >>> inside string
def DCGM_STR_IS_BLANK(val):
    if 0 != val.find("<<<"):
        return False
    elif 0 > val.find(">>>"):
        return False
    return True

###############################################################################
class DcgmValue:
    def __init__(self, value):
        self.value = value #Contains either an integer (int64), string, or double of the actual value

    ###########################################################################
    def SetFromInt32(self, i32Value):
        '''
        Handle the special case where our source data was an int32 but is currently
        stored in a python int (int64), dealing with blanks
        '''
        value = int(i32Value)

        if not DCGM_INT32_IS_BLANK(i32Value):
            self.value = value
            return

        if value == DCGM_INT32_NOT_FOUND:
            self.value = DCGM_INT64_NOT_FOUND
        elif value == DCGM_INT32_NOT_SUPPORTED:
            self.value = DCGM_INT64_NOT_SUPPORTED
        elif value == DCGM_INT32_NOT_PERMISSIONED:
            self.value = DCGM_INT64_NOT_PERMISSIONED
        else:
            self.value = DCGM_INT64_BLANK

    ###########################################################################
    def IsBlank(self):
        '''
        Returns True if the currently-stored value is a blank value. False if not
        '''
        if self.value is None:
            return True
        elif type(self.value) == int or type(self.value) == int:
            return DCGM_INT64_IS_BLANK(self.value)
        elif type(self.value) == float:
            return DCGM_FP64_IS_BLANK(self.value)
        elif type(self.value) == str:
            return DCGM_STR_IS_BLANK(self.value)
        else:
            raise Exception("Unknown type: %s") % str(type(self.value))

    ###########################################################################
    def __str__(self):
        return str(self.value)

    ###########################################################################

###############################################################################
def self_test():

    v = DcgmValue(1.0)
    assert(not v.IsBlank())
    assert(v.value == 1.0)

    v = DcgmValue(100)
    assert(not v.IsBlank())
    assert(v.value == 100)

    v = DcgmValue(DCGM_INT64_NOT_FOUND)
    assert(v.IsBlank())

    v = DcgmValue(DCGM_FP64_NOT_FOUND)
    assert(v.IsBlank())

    v.SetFromInt32(DCGM_INT32_NOT_SUPPORTED)
    assert(v.IsBlank())
    assert(v.value == DCGM_INT64_NOT_SUPPORTED)

    print("Tests passed")
    return

###############################################################################
if __name__ == "__main__":
    self_test()

###############################################################################


