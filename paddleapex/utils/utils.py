# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import time
import os
import paddle
from importlib import import_module


def save_tensor(tensor, file_path):
    paddle.save(tensor, file_path)
    print(file_path, " Finished!")


ctx = multiprocessing.get_context("fork")


def try_import(moduleName="paddle"):
    try:
        MODULE = import_module(moduleName)
        # globals()[moduleName]
        return moduleName, MODULE
    except ImportError as err:
        print(f"Import {moduleName} failed, error message is {err}")
        return None, None


class ThreadPool:
    def __init__(self, max_process_num=3) -> None:
        self.max_process_num = max_process_num
        self.event_queue = []

    def __del__(self):
        self.async_exit()

    def safe_parellel_save(self, tensor, file_path, remote_path):
        self.allocate_subprocess()
        name = file_path.split("/")
        remote_path = os.path.join(remote_path, name[-1])
        print(f"Async save tensor:{remote_path}")

        p = ctx.Process(target=save_tensor, args=(tensor, remote_path))
        p.start()
        self.event_queue.append(p)

    def allocate_subprocess(self):
        """
        wait until all async save task to be done.
        """
        while len(self.event_queue) > self.max_process_num:
            task = self.event_queue.pop()
            if task and task.is_alive():
                if task.is_alive():
                    self.event_queue.append(task)
                    continue
        if len(self.event_queue) < self.max_process_num:
            return
        else:
            while len(self.event_queue) == self.max_process_num:
                task = self.event_queue.pop()
                if task and task.is_alive():
                    if task.is_alive():
                        self.event_queue.append(task)
                        time.sleep(0.5)
                        continue

    def async_exit(self):
        while len(self.event_queue) > 0:
            task = self.event_queue.pop()
            if task and task.is_alive():
                if task.is_alive():
                    self.event_queue.append(task)
                    print("waiting.....")
                    time.sleep(0.5)
                    continue
        print("Sub processes have done, async process exit.")
