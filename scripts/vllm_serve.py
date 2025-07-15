# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import argparse
import asyncio
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch


from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)

if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI

if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port
    from ..scripts.vllm_patch import (
    LoRARequest as PatchedLoRARequest,
    WorkerLoRAManager as PatchedWorkerLoRAManager,
    LRUCacheWorkerLoRAManager as PatchedLRUCacheWorkerLoRAManager,
    )
    from vllm.lora.request import LoRARequest
    def patch():
        import vllm.lora.request
        vllm.lora.request.LoRARequest = PatchedLoRARequest
        import vllm.lora.worker_manager
        vllm.lora.worker_manager.LoRARequest = PatchedLoRARequest
        vllm.lora.worker_manager.WorkerLoRAManager = PatchedWorkerLoRAManager
        vllm.lora.worker_manager.LRUCacheWorkerLoRAManager = PatchedLRUCacheWorkerLoRAManager
    patch()
    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 1. 将函数定义放在这里
def _worker_generate_with_lora(worker, prompts, sampling_params):
    """
    A top-level function to be executed on each worker via collective_rpc.
    'worker' is the vLLM worker instance in the background process.
    """
    # worker 对象有 worker_extension 属性，里面有我们自定义的方法
    return worker.worker_extension.generate_with_lora(
        prompts=prompts,
        sampling_params=sampling_params
    )
class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """
    def __init__(self, *args, **kwargs):
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )

        super().__init__(*args, **kwargs)

        # The following attributes are initialized when `init_communicator` method is called.
        self.pynccl_comm = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights
        self.lora_weight = {}
        self.lora_requests = None
        self.lora_id=0
    # Class-level attributes to ensure they exist on the worker instance
    pynccl_comm = None
    client_rank = None
    lora_weight = {}
    lora_requests = None
    lora_id = 0


    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to communicate with vLLM
        workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`str`):
                Data type of the weight tensor as a string (e.g., `"torch.float32"`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def update_lora_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.lora_weight[name]=weight
     

    def _convert_and_merge_lora_weights(self) -> dict:
        """
        最终解决方案 V10 (极简版): 
        假设接收到的权重名称是干净的，形状是正确的。
        此函数只负责将分离的 q,k,v 和 gate,up 权重合并。
        """
        unwrapped_weights = {}
        logger.info("Merging received LoRA weights for vLLM (V10)...")
        
        # 我们直接使用 self.lora_weight
        module_to_weights = self.lora_weight
        
        processed_modules = set()
        # 注意：遍历 module_to_weights 的键，因为它们是干净的名称
        for module_name in list(module_to_weights.keys()):
            if module_name in processed_modules: continue

            # --- QKV 合并逻辑 ---
            if module_name.endswith(".q_proj.lora_A.weight"):
                base_path = module_name.replace(".q_proj.lora_A.weight", "")
                q_A_name, q_B_name = module_name, module_name.replace("lora_A", "lora_B")
                k_A_name, k_B_name = f"{base_path}.k_proj.lora_A.weight", f"{base_path}.k_proj.lora_B.weight"
                v_A_name, v_B_name = f"{base_path}.v_proj.lora_A.weight", f"{base_path}.v_proj.lora_B.weight"

                # 检查所有 q,k,v 的 A,B 权重是否存在
                all_qkv_keys = [q_A_name, q_B_name, k_A_name, k_B_name, v_A_name, v_B_name]
                if all(key in module_to_weights for key in all_qkv_keys):
                    logger.info(f"Processing QKV group for: {base_path}")
                    
                    vllm_module_name = f"{base_path}.qkv_proj"
                    # QKV的lora_A是共享的
                    unwrapped_weights[f"{vllm_module_name}.lora_A.weight"] = module_to_weights[q_A_name]
                    # 合并LoRA B
                    unwrapped_weights[f"{vllm_module_name}.lora_B.weight"] = torch.cat([
                        module_to_weights[q_B_name],
                        module_to_weights[k_B_name],
                        module_to_weights[v_B_name]
                    ], dim=0)
                    
                    processed_modules.update(all_qkv_keys)
                    continue

            # --- Gate/Up 合并逻辑 ---
            if module_name.endswith(".gate_proj.lora_A.weight"):
                base_path = module_name.replace(".gate_proj.lora_A.weight", "")
                g_A_name, g_B_name = module_name, module_name.replace("lora_A", "lora_B")
                u_A_name, u_B_name = f"{base_path}.up_proj.lora_A.weight", f"{base_path}.up_proj.lora_B.weight"
                
                all_gu_keys = [g_A_name, g_B_name, u_A_name, u_B_name]
                if all(key in module_to_weights for key in all_gu_keys):
                    logger.info(f"Processing Gate/Up group for: {base_path}")
                    
                    vllm_module_name = f"{base_path}.gate_up_proj"
                    # 合并 LoRA A 和 B
                    unwrapped_weights[f"{vllm_module_name}.lora_A.weight"] = torch.cat([module_to_weights[g_A_name], module_to_weights[u_A_name]], dim=0)
                    unwrapped_weights[f"{vllm_module_name}.lora_B.weight"] = torch.cat([module_to_weights[g_B_name], module_to_weights[u_B_name]], dim=0)
                    
                    processed_modules.update(all_gu_keys)
                    continue

        # 处理所有未被合并的独立模块
        for module_name, weight in module_to_weights.items():
            if module_name not in processed_modules:
                logger.info(f"Keeping independent module: {module_name}")
                unwrapped_weights[module_name] = weight
        
        logger.info(f"Weight conversion complete. Final LoRA modules for vLLM: {len(unwrapped_weights) // 2}")
        # 4. 最终健全性检查
        #     logger.info("--- Running Final Sanity Checks ---")
        return unwrapped_weights

     
    def apply_lora(self, lora_config: dict) -> int:
        """使用收集到的LoRA权重创建并注册LoRA适配器"""
        #from vllm.lora.request import LoRARequest
        # for name, weight_param in self.lora_weight.items():
        #     print(name)
        # 创建唯一的LoRA ID
        self.lora_id += 1
        self.lora_config = lora_config
        # 处理FSDP包装的参数
        #processed_weights = self._convert_fsdp_parameters()
        # 创建LoRA请求对象
        lora_request = PatchedLoRARequest(
            lora_name=f"lora_{self.lora_id}",
            lora_int_id=self.lora_id,
            lora_tensors=self.lora_weight,#processed_weights self.lora_weight
            lora_config=lora_config,
            #lora_path=f"lora_{self.lora_id}"
        )
        self.lora_requests = lora_request
        self.add_lora(self.lora_requests)
        return self.lora_id
    
    def get_lora_request(self):
        """一个简单的 RPC 目标，返回缓存的 LoRA 请求对象。"""
        # from vllm.lora.request import LoRARequest
        # # 处理FSDP包装的参数
        # # 创建LoRA请求对象
        # lora_request = LoRARequest(
        #     lora_name=f"lora_{self.lora_id}",
        #     lora_int_id=self.lora_id,
        #     lora_tensors=self.lora_weight, ##processed_weights self.lora_weight
        #     lora_config=self.lora_config,
        # )
        return self.lora_requests

    # 新增这个方法，只返回安全的、可序列化的信息
    def get_lora_info(self) -> Optional[dict]:
        """
        返回当前激活的 LoRA 的基本信息（名称和 ID），这是一个可安全序列化的字典。
        """
        if self.lora_requests:
            return {
                "lora_name": self.lora_requests.lora_name,
                "lora_int_id": self.lora_requests.lora_int_id,
                "lora_path": self.lora_requests.lora_name,
            }
        return None

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool`, *optional*, defaults to `False`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading models. Set to `True` to allow executing code from model
            repositories. This is required for some custom models but introduces security risks.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models. Set to True to allow executing code from model "
            "repositories. This is required for some custom models but introduces security risks."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    # --- 核心修复在这里 ---
    enable_lora: bool = field(
        default=False,
        metadata={
            "help": "Enable LoRA support in vLLM." # 使用 'key': 'value' 格式
        },
    )
    max_lora_rank: int = field(
        default=512,
        metadata={
            "help": "Maximum rank for LoRA adapters." # 使用 'key': 'value' 格式
        },
    )
    max_cpu_loras: int = field(
        default=1,
        metadata={
            "help": "Maximum number of LoRA adapters to be stored in CPU memory." # 使用 'key': 'value' 格式
        },
    )

def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_lora=script_args.enable_lora,
        max_lora_rank=script_args.max_lora_rank if script_args.enable_lora else None,
        max_cpu_loras=script_args.max_cpu_loras if script_args.enable_lora else None,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
        trust_remote_code=script_args.trust_remote_code,
    )
    # print(f"--- [DP RANK {data_parallel_rank}] DIAGNOSTICS ---")
    # print(f"llm object type: {type(llm)}")
    # print(f"llm_engine object type: {type(llm.llm_engine.engine_core)}")
    # print(f"Attributes of llm_engine: {dir(llm.llm_engine.engine_core)}")
    # print(f"--- END DIAGNOSTICS ---")
    # ========================== 修改结束 ==========================
    # Send ready signal to parent process
    connection.send({"status": "ready"})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break
        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
             # --- 关键的拦截和转换逻辑 ---
            if method_name == "generate" and "lora_request" in kwargs:
                lora_dict = kwargs.get("lora_request")
                # 如果收到了字典，就在 worker 内部将其转换为 LoRARequest 对象
                if isinstance(lora_dict, dict):
                    kwargs["lora_request"] = PatchedLoRARequest(
                        lora_name=lora_dict["lora_name"],
                        lora_int_id=lora_dict["lora_int_id"],
                        lora_path=lora_dict["lora_path"]
                    )
                # 如果 lora_dict 是 None，则保持为 None，vLLM 会正确处理
            method = getattr(llm, method_name)
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
    ```python
    >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
    [[1, 2, 3], [4, 5, 6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
    [[1, 2], [3, 4], [5], [6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
    [[1], [2], [3], [4], [5], [6], [], []]
    ```
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f"Process {process} is still alive after 10 seconds, attempting to terminate...")
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    app = FastAPI(lifespan=lifespan)

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        generation_kwargs: dict = field(default_factory=dict)

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.
                - `n` (`int`, *optional*, defaults to `1`): Number of completions to generate for each prompt.
                - `repetition_penalty` (`float`, *optional*, defaults to `1.0`): Repetition penalty to apply during generation.
                - `temperature` (`float`, *optional*, defaults to `1.0`): Temperature for sampling. Higher values lead to more random outputs.
                - `top_p` (`float`, *optional*, defaults to `1.0`): Top-p (nucleus) sampling parameter. It controls the diversity of the generated text.
                - `top_k` (`int`, *optional*, defaults to `-1`): Top-k sampling parameter. If set to `-1`, it disables top-k sampling.
                - `min_p` (`float`, *optional*, defaults to `0.0`): Minimum probability threshold for sampling.
                - `max_tokens` (`int`, *optional*, defaults to `16`): Maximum number of tokens to generate for each completion.
                - `guided_decoding_regex` (`str`, *optional*): A regex pattern for guided decoding. If provided, the model will only generate tokens that match this regex pattern.
                - `generation_kwargs` (`dict`, *optional*): Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they will override them.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "guided_decoding": guided_decoding,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]

            if script_args.enable_lora:
                # ======================= 核心修改在这里 =======================
                # 1. 通过 RPC 从 worker 获取 LoRA 的基本信息（字典）
                get_lora_info_command = {
                    "type": "call",
                    "method": "collective_rpc",
                    "kwargs": {"method": "get_lora_info"}
                }
                connection.send(get_lora_info_command)
                # 使用 asyncio.to_thread 异步等待阻塞的 recv 调用
                lora_info_results = await asyncio.to_thread(connection.recv)

                lora_request_dict = None
                # 我们只需要第一个 worker 的信息
                if lora_info_results and lora_info_results[0]:
                    lora_request_dict = lora_info_results[0]
                    # vLLM 的 generate 方法需要一个 lora_path
                    # 即使我们从内存加载，也需要提供一个虚拟路径
                    if "lora_path" not in lora_request_dict:
                        lora_request_dict["lora_path"] = lora_request_dict["lora_name"]
        
                kwargs = {"prompts": prompts, "sampling_params": sampling_params, "lora_request": lora_request_dict}
            else:
                kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            #rpc_args = (prompts, sampling_params)
            #kwargs = {"method": "generate_with_lora",'args': rpc_args}
            #rpc_args = (prompts, sampling_params)
            #kwargs = {"method": "generate_with_lora",'args': rpc_args}
            #connection.send({"type": "call", "method": "collective_rpc", "kwargs": kwargs})
            #kwargs = {"prompts": prompts, "sampling_params": sampling_params, "lora_request": final_lora_request}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})
        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, world_size)}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function update_named_param is called this way: update_named_param("name", "torch.float32", (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        kwargs = {"method": "update_named_param", "args": (request.name, request.dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, updating named parameter"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_lora_param/")
    async def update_lora_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
         # The function update_named_param is called this way: update_named_param("name", "torch.float32", (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        kwargs = {"method": "update_lora_param", "args": (request.name, request.dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, updating update_lora_param parameter"}
    
    # --- 开始修改 ---
    # 移除旧的 apply_lora 端点，替换为下面的正确实现
    class ApplyLoraRequest(BaseModel):
        lora_config: dict

    @app.post("/apply_lora/")
    async def apply_lora(request: ApplyLoraRequest):
        """
        Applies the collected LoRA weights to the model across all workers.

        This endpoint triggers all vLLM workers to create and register a new
        LoRA adapter using the weights previously sent via 'apply_lora'.

        Args:
            request (`ApplyLoraRequest`):
                - `lora_config` (`dict`): The configuration dictionary for the LoRA adapter.
        
        Returns:
            A dictionary containing the message and the new LoRA ID if successful,
            or an error message.
        """
        # 1. 构造发送给 worker 的命令
        # 我们要调用 worker 内部 llm 对象的 apply_lora 方法
        kwargs = {"method": "apply_lora", "args": (request.lora_config,)}
        # 2. 通过 Pipe 连接将命令发送给所有 worker 进程
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})


        return {"message": "Request received, apply_lora sussessed!"}
    

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        # Wait for and collect all results
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
