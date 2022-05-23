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

from collections import defaultdict
import itertools

from .record import Record
from .da_exceptions import TorchBenchAnalyzerException


class RecordAggregator:
    """
    Stores a collection of Record objects.
    """

    def __init__(self):
        self._records = defaultdict(list)

    def insert(self, record):
        """
        Insert a record into the RecordAggregator

        Parameters
        ----------
        record : Record
            A record to be inserted
        """

        if isinstance(record, Record):
            record_type = type(record)
            self._records[record_type].append(record)
        else:
            raise TorchBenchAnalyzerException(
                "Can only add objects of type 'Record' to RecordAggregator")

    def insert_all(self, record_list):
        """
        Insert records from a list of records
        into the RecordAggregator

        Parameters
        ----------
        record_list : List of Records
            The records to insert
        """

        for record in record_list:
            self.insert(record)

    def add_key(self, record_type, records):
        """
        Adds or replaces all the records of a given record_type with the new
        records

        Parameters
        ----------
        record_type : Record
            record_type to add to the records.
        records : list
            List of new records to be added.
        """

        self._records[record_type] = records

    def filter_records(self, record_types=None, filters=None):
        """
        Get records that satisfy the given list of criteria.

        Parameters
        ----------

        record_types : list of types of Records
            the types of the records we are
            imposing the filter criteria on.

        filters : list of callables
            conditions that determine whether
            a given record should be returned.
            If no filters specified, all records
            of types specified by record_types will be
            returned.
            Note : This must be of the same length
                   as the list of record_types, or omitted.

        Returns
        -------
        RecordAggregator
            Returns a new RecordAggregator containing the filtered
            records
        """

        filtered_records = RecordAggregator()
        if not record_types and not filters:
            for record_type, records in self._records.items():
                filtered_records.add_key(record_type, records)
            return filtered_records

        if record_types and not filters:
            try:
                for record_type in record_types:
                    filtered_records.add_key(record_type,
                                             self._records[record_type])
                return filtered_records
            except KeyError as k:
                raise TorchBenchAnalyzerException(
                    f"Record type '{k.header()}' not found in this RecordAggregator"
                )
        if filters and not record_types:
            raise TorchBenchAnalyzerException(
                "Must specify the record types corresponding to each filter criterion."
            )
        if len(record_types) != len(filters):
            raise TorchBenchAnalyzerException(
                "Must specify the same number of record types as filter criteria."
            )

        # Remove records that do not satisfy criteria
        for h, f in zip(record_types, filters):
            for record in self._records[h]:
                if f(record):
                    filtered_records.insert(record)

        return filtered_records

    def groupby(self, record_types, groupby_criterion):
        """
        Group all the records of a certain type together if they have the
        same value for a given groupbby criteria.

        Parameters
        ----------
        record_types : list
            A list of record type
        groupby_criterion : callable
            This callable will receive a single record as the argument and
            must return the value that will be used for groupby

        Returns
        -------
        dict
            A dictionary of dictionaries where the first level keys are the
            record type and the second level keys are unique values returned
            by groupby_criteria and the values are the aggregated records.
        """

        field_values = {
            record_type: set([
                groupby_criterion(record)
                for record in self._records[record_type]
            ]) for record_type in record_types
        }
        groupby_result = defaultdict(list)
        for record_type in record_types:
            groupby_result[record_type] = defaultdict(list)
            for field_value in field_values[record_type]:
                aggregated_result = self.filter_records(
                    record_types=[record_type],
                    filters=[lambda r: groupby_criterion(r) == field_value
                            ]).aggregate(record_types=[record_type])
                groupby_result[record_type][field_value] = \
                    aggregated_result[record_type]
        return groupby_result

    def record_types(self):
        """
        Returns
        -------
        list of str
            a list of the types of records in this
            RecordAgrregator
        """

        return list(self._records)

    def total(self, record_type=None):
        """
        Get the total number of records in
        the RecordAggregator

        Parameters
        ----------
        record_type : a class name of type Record
            The type of records to count,
            if None, count all types

        Returns
        -------
        int
            number of records in
            the RecordAggregator
        """

        if record_type:
            if record_type not in self._records:
                raise TorchBenchAnalyzerException(
                    f"Record type '{record_type.header()}' not found in this RecordAggregator"
                )
            return len(self._records[record_type])
        return sum(len(self._records[k]) for k in self._records)

    def aggregate(self, record_types=None):
        """
        Parameters
        ----------
        record_types : List of Record types
            The type of records to aggregate.
            If None, aggregates all records

        Returns
        -------
        dict
            keys are requested record types
            and values are the aggregated values
        """

        if not record_types:
            record_types = self.record_types()
        aggregated_records = {
            record_type:
            record_type.aggregation_function()(self._records[record_type])
            for record_type in record_types
        }
        return aggregated_records

    def get_records(self):
        """
        Get all the records.

        Returns
        -------
        dict
            A dictionary where the keys are record types and the values are
            an array of records with the specified type
        """

        return self._records

    def _flatten_records(self, records):
        """
        Flatten the records array by joining all the arrays together.
        """

        return list(itertools.chain.from_iterable(records))
