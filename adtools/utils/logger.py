"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import json
import os
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Optional


class ArbitraryJsonLogger:
    def __init__(self, logdir: PathLike | str, file_name: str, write_file_frequency: Optional[int] = 100):
        """Log something to {logdir}/{file_name}.json
        """
        self._counter = 1
        self._logdir = Path(logdir)
        self._file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
        self._write_file_frequency = write_file_frequency
        self._write_cache_lock = Lock()
        self._write_file_lock = Lock()

        # Create an empty json file
        os.makedirs(self._logdir, exist_ok=True)
        with open(self._logdir / self._file_name, 'w') as f:
            json.dump([], f)

        self._cache = []

    def log_to_cache(self, json_dict: dict):
        with self._write_cache_lock:
            json_dict['count'] = self._counter
            self._cache.append(json_dict)
            if self._write_file_frequency and self._counter % self._write_file_frequency == 0:
                self.write_cache_to_file()
            self._counter += 1

    def write_cache_to_file(self):
        with self._write_file_lock:
            with open(self._logdir / self._file_name, 'r') as f:
                data = json.load(f)
            data.extend(self._cache)
            with open(self._logdir / self._file_name, 'w') as f:
                json.dump(data, f)
            self._cache = []

    def __del__(self):
        self.write_cache_to_file()
