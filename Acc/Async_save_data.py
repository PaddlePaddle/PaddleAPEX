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
import paddle
import multiprocessing
import time

def save_tensor(tensor_np, file_path):
    tensor = paddle.to_tensor(tensor_np)
    paddle.save(tensor, file_path)
    print(file_path," Finished!")

class ThreadPool:
    def __init__(self, max_process_num=3) -> None:
        self.max_process_num = max_process_num

    def __del__(self):
        self.async_exit()

    def safe_parellel_save(self, tensor_np, file_path):
        launcher = multiprocessing.get_context("spawn")
        self.allocate_subprocess()
        sub_process = launcher.Process(target=save_tensor,args=(tensor_np, file_path))
        print(f"Async save tensor:{file_path}")
        sub_process.start()
        event_queue.append(sub_process)

    def allocate_subprocess(self):
        """
        wait until all async save task to be done.
        """
        while len(event_queue) > self.max_process_num:
            task = event_queue.pop()
            if task and task.is_alive():
                if task.is_alive():
                    event_queue.append(task)
                    continue
        if len(event_queue) < self.max_process_num:
            return
        else:
            while len(event_queue) == self.max_process_num:
                task = event_queue.pop()
                if task and task.is_alive():
                    if task.is_alive():
                        event_queue.append(task)
                        time.sleep(1)
                        continue

    def async_exit(self):
        while len(event_queue) > 0:
            task = event_queue.pop()
            if task and task.is_alive():
                if task.is_alive():
                    event_queue.append(task)
                    print("waiting.....")
                    time.sleep(2)
                    continue
        print("Sub processes have done, async process exit.")

event_queue=[]