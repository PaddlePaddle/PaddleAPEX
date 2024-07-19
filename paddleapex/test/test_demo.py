"""
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
"""
import paddle
from paddleapex import Tracer

if __name__ == "__main__":
    a = paddle.randn([2,2])
    b = paddle.randn([2,2])
    a.stop_gradient = False
    b.stop_gradient = False
    apex = Tracer()
    apex.start()
    y = paddle.add(a,b)
    y = paddle.multiply(y,b)
    z = y ** 2
    z.backward()
    apex.stop()