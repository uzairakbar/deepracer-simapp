#################################################################################
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          #
#                                                                               #
#   Licensed under the Apache License, Version 2.0 (the "License").             #
#   You may not use this file except in compliance with the License.            #
#   You may obtain a copy of the License at                                     #
#                                                                               #
#       http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                               #
#   Unless required by applicable law or agreed to in writing, software         #
#   distributed under the License is distributed on an "AS IS" BASIS,           #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#   See the License for the specific language governing permissions and         #
#   limitations under the License.                                              #
#################################################################################

'''This module implement deepracer boto client'''

import os
import time
import random
import shutil
import logging
import botocore
import boto3

from markov.log_handler.logger import Logger
from markov.constants import (NUM_RETRIES, CONNECT_TIMEOUT)
from markov.boto.constants import BOTO_ERROR_MSG_FORMAT

LOG = Logger(__name__, logging.INFO).get_logger()


class BotoClientLocalFileSystemWrapper(object):
    """Local-filesystem stand-in for a boto S3 client.

    S3 operations are mapped to files under ``/<Bucket>/<Key>`` on the container
    filesystem. This is a port of the legacy ``uzairakbar/deepracer:v0`` shim:
    the standalone sim has no real AWS and does not need an in-container MinIO
    HTTP server -- removing that round-trip eliminates the truncated-download
    crashes seen on shared HPC nodes. Only the subset of the boto S3 client API
    that ``markov`` actually calls is implemented (download/upload/put/delete/
    list/paginate).
    """

    @staticmethod
    def _local_path(bucket, key):
        # A leading '/' on key would make os.path.join drop the bucket, so strip
        # it: keys are always relative to the bucket root.
        return os.path.join("/", bucket, str(key).lstrip("/"))

    @staticmethod
    def _ensure_parent(path):
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def download_file(self, Bucket, Key, Filename, **kwargs):
        src = self._local_path(Bucket, Key)
        if not os.path.isfile(src):
            # Mimic boto's missing-key behaviour: real S3 raises ClientError(404)
            # for an absent object, and callers rely on that to treat optional
            # files as "not present" (e.g. download_custom_files_if_present).
            # Raising FileNotFoundError instead would be caught as a fatal error.
            raise botocore.exceptions.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "GetObject")
        self._ensure_parent(Filename)
        shutil.copyfile(src, Filename)

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, **kwargs):
        dst = self._local_path(Bucket, Key)
        self._ensure_parent(dst)
        shutil.copyfile(Filename, dst)

    def upload_fileobj(self, Fileobj, Bucket, Key, ExtraArgs=None, **kwargs):
        dst = self._local_path(Bucket, Key)
        self._ensure_parent(dst)
        with open(dst, "wb") as outfile:
            outfile.write(Fileobj.getbuffer())

    def put_object(self, **kwargs):
        dst = self._local_path(kwargs["Bucket"], kwargs["Key"])
        self._ensure_parent(dst)
        body = kwargs.get("Body", b"")
        if isinstance(body, str):
            body = body.encode()
        with open(dst, "wb") as outfile:
            outfile.write(body)

    def delete_object(self, **kwargs):
        dst = self._local_path(kwargs["Bucket"], kwargs["Key"])
        if os.path.exists(dst):
            os.remove(dst)

    def _list_keys(self, bucket, prefix):
        base = self._local_path(bucket, prefix)
        keys = []
        if os.path.isdir(base):
            for fname in sorted(os.listdir(base)):
                if os.path.isfile(os.path.join(base, fname)):
                    keys.append(os.path.join(str(prefix).lstrip("/"), fname))
        return keys

    def list_objects_v2(self, **kwargs):
        keys = self._list_keys(kwargs["Bucket"], kwargs.get("Prefix", ""))
        if not keys:
            return {}
        return {"Contents": [{"Key": key} for key in keys]}

    def paginate(self, **kwargs):
        keys = self._list_keys(kwargs["Bucket"], kwargs.get("Prefix", ""))
        # single page, mirroring a boto page iterator (Contents always a list)
        return [{"Contents": [{"Key": key} for key in keys]}]

    def get_paginator(self, operation_name):
        return self


class DeepRacerBotoClient(object):
    """Deepracer boto client class
    """
    def __init__(self, region_name='us-east-1', 
                 s3_endpoint_url=None,
                 max_retry_attempts=5,
                 backoff_time_sec=1.0, boto_client_name=None,
                 session=None):
        """Deepracer boto client class

        Args:
          region_name (str): aws region name
          max_retry_attempts (int): max retry attempts for client call
          backoff_time_sec (float): exp back off time between call
          boto_client_name (str): boto client name
          session (boto3.Session): An alternative session to use.
                                   Defaults to None.
        """
        self._region_name = region_name
        self._s3_endpoint_url = s3_endpoint_url        
        self._max_retry_attempts = max_retry_attempts
        self._backoff_time_sec = backoff_time_sec
        self._boto_client_name = boto_client_name
        self._session = session

    def _get_boto_config(self):
        """Returns a botocore config object which specifies the number of times to retry"""

        return botocore.config.Config(retries=dict(max_attempts=NUM_RETRIES),
                                      connect_timeout=CONNECT_TIMEOUT)

    def _get_client(self):
        """Return the local-filesystem S3 shim (no boto/MinIO).

        The standalone sim has no real AWS; S3 keys are served from the local
        filesystem under /<bucket>/<key>. See BotoClientLocalFileSystemWrapper.
        """
        return BotoClientLocalFileSystemWrapper()

    def get_client(self):
        """Return boto client with backoff retry logic"""
        return self.exp_backoff(self._get_client)

    def exp_backoff(self, action_method, **kwargs):
        """retry on action_method

        Args:
            action_method (method) : specific action method
            **kwargs: argument for action_method
        """

        # download with retry
        try_count = 0
        while True:
            try:
                return action_method(**kwargs)
            except Exception as e:
                try_count += 1
                if try_count > self._max_retry_attempts:
                    raise e
                # use exponential backoff
                backoff_time = (pow(try_count, 2) + random.random()) * self._backoff_time_sec
                error_message = BOTO_ERROR_MSG_FORMAT.format(self._boto_client_name,
                                                             backoff_time,
                                                             str(try_count),
                                                             str(self._max_retry_attempts),
                                                             e)
                LOG.info(error_message)
                time.sleep(backoff_time)
