"""
Date: 2022-07-18 2:15:47 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-09-14 3:44:19 pm
LastEditors: dihuangdh
"""

import json
import pickle
import warnings
from io import BytesIO, StringIO  # TODO:
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Tuple, Union

import cv2
import numpy as np


def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.
    Args:
        method (str): The method name to check.
        obj (object): The object to check.
    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


class PetrelBackend:
    """Petrel storage backend - simple version"""

    def __init__(self, enable_mc: bool = False) -> None:
        try:
            from petrel_client.client import Client
        except ImportError:
            raise ImportError(
                "Please install petrel_client to enable " "PetrelBackend."
            )

        self._client = Client(enable_mc=enable_mc)

    def get(self, filepath) -> memoryview:
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self, filepath, warning=False) -> str:
        try:
            value = self._client.Get(filepath)
        except:
            if warning:
                warning.warn("Failed to get text from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get text from {}".format(filepath))
        return str(value, encoding="utf-8")

    def get_uint16_png(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint8)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except:
            if warning:
                warning.warn("Failed to get uint16_png from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get uint16_png from {}".format(filepath))
        return value

    def get_uint8_jpg(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint8)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except:
            if warning:
                warning.warn("Failed to get uint8_jpg from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get uint8_jpg from {}".format(filepath))
        return value

    def get_numpy_array(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = BytesIO(value)
            value = np.load(value)
        except:
            if warning:
                warning.warn("Failed to get npz from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get npz from {}".format(filepath))
        return value

    def get_npz(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = np.loads(value)
        except:
            if warning:
                warning.warn("Failed to get npz from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get npz from {}".format(filepath))
        return value

    def get_numpy_txt(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.loadtxt(StringIO(self.get_text(filepath)))
        except:
            if warning:
                warning.warn("Failed to get numpy_txt from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get numpy_txt from {}".format(filepath))
        return value

    def get_json(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = json.loads(value)
        except:
            if warning:
                warning.warn("Failed to get json from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get json from {}".format(filepath))
        return value

    def put_uint16_png(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".png", value, params=[cv2.CV_16U])
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_uint8_jpg(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".jpg", value)
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_npz(self, filepath, value) -> None:
        value = pickle.dumps(value)
        self._client.put(filepath, value)
        # self._client.put(filepath, value, update_cache=True)

    def put_json(self, filepath, value) -> None:
        value = json.dumps(value).encode()
        self._client.put(filepath, value)
        # self._client.put(filepath, value, update_cache=True)

    def put_text(self, filepath, value) -> None:
        self._client.put(filepath, bytes(value, encoding="utf-8"))
        # self._client.put(filepath, bytes(value, encoding='utf-8'), update_cache=True)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        """Concatenate all file paths.
        Args:
            filepath (str or Path): Path to be concatenated.
        Returns:
            str: The result after concatenation.
        """
        # filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith("/"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_paths.append(path)
        return "/".join(formatted_paths)

    # from mmcv
    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        # if not has_method(self._client, 'list'):
        #     raise NotImplementedError(
        #         'Current version of Petrel Python SDK has not supported '
        #         'the `list` method, please use a higher version or dev'
        #         ' branch instead.')

        # dir_path = self._map_path(dir_path)
        # dir_path = self._format_path(dir_path)
        # if list_dir and suffix is not None:
        #     raise TypeError(
        #         '`list_dir` should be False when `suffix` is not None')

        # if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        #     raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith("/"):
            dir_path += "/"

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith("/"):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root) : -1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            next_dir_path, list_dir, list_file, suffix, recursive
                        )
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root) :]
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)

    # from mmcv
    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.
        Args:
            filepath (str or Path): Path to be checked whether exists.
        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        if not (
            has_method(self._client, "contains") and has_method(self._client, "isdir")
        ):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` and `isdir` methods, please use a higher"
                "version or dev branch instead."
            )

        return self._client.contains(filepath) or self._client.isdir(filepath)
