# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Base exception class
class QuantMasterException(Exception):
    pass


class RecorderInitializationError(QuantMasterException):
    """Error type for re-initialization when starting an experiment"""


class LoadObjectError(QuantMasterException):
    """Error type for Recorder when can not load object"""


class ExpAlreadyExistError(Exception):
    """Experiment already exists"""
