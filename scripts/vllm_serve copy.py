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

    # def _convert_fsdp_parameters(self, lora_config: dict) -> dict:
    #     """
    #     转换FSDP权重为vLLM兼容的LoRA权重格式。
    #     V2: 采用“聚合-切分”策略，并增强了模块匹配逻辑和诊断日志。
    #     """
    #     unwrapped_weights = {}
        
    #     # 1. 从LoRA配置中获取关键信息
    #     rank = lora_config.get("r")
    #     if rank is None:
    #         raise ValueError("LoRA configuration must contain the rank 'r'.")
        
    #     target_modules_set = set(lora_config.get("target_modules", []))
    #     if not target_modules_set:
    #          raise ValueError("LoRA configuration must contain 'target_modules'.")

    #     # 2. 聚合所有收到的FSDP展平参数
    #     flat_params_A = {}
    #     flat_params_B = {}
    #     for name, weight_param in self.lora_weight.items():
    #         if ".lora_A." in name:
    #             flat_params_A[name] = weight_param
    #         elif ".lora_B." in name:
    #             flat_params_B[name] = weight_param
        
    #     # 按名称排序并拼接，以模拟FSDP的参数顺序
    #     sorted_A_tensors = [flat_params_A[k] for k in sorted(flat_params_A.keys())]
    #     sorted_B_tensors = [flat_params_B[k] for k in sorted(flat_params_B.keys())]

    #     if not sorted_A_tensors or not sorted_B_tensors:
    #         logger.warning("No LoRA A or B weights found to convert.")
    #         return {}

    #     all_A_weights = torch.cat([t.flatten() for t in sorted_A_tensors])
    #     all_B_weights = torch.cat([t.flatten() for t in sorted_B_tensors])

    #     logger.info(f"Aggregated all LoRA A weights into a single tensor of size: {all_A_weights.numel()}")
    #     logger.info(f"Aggregated all LoRA B weights into a single tensor of size: {all_B_weights.numel()}")

    #     offset_A, offset_B = 0, 0

    #     # 3. 遍历模型的目标模块，按顺序切分并重塑权重
    #     # 关键：迭代基础模型的模块以获得正确的顺序和维度
    #     for module_name, module in self.model_runner.model.named_modules():
            
    #         # --- 核心修改在这里 ---
    #         # 3.1 首先，检查模块是否是我们可以应用LoRA的类型
    #         if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
    #             continue

    #         # 3.2 然后，检查该模块的名称是否是我们的目标之一
    #         # 我们检查模块名的最后一部分 (e.g., 'q_proj')
    #         module_key = module_name.split('.')[-1]
    #         if module_key not in target_modules_set:
    #             logger.debug(f"Skipping module '{module_name}' (key '{module_key}' not in target set).")
    #             continue
            
    #         logger.info(f"Found target module: '{module_name}' (type: {type(module).__name__})")
            
    #         # 3.3 确定层的输入和输出维度
    #         in_features, out_features = None, None
    #         if isinstance(module, torch.nn.Linear):
    #             in_features, out_features = module.in_features, module.out_features
    #         elif isinstance(module, torch.nn.Embedding):
    #             in_features, out_features = module.num_embeddings, module.embedding_dim

    #         # 计算此模块LoRA矩阵所需的元素数量
    #         num_elements_A = rank * in_features
    #         num_elements_B = out_features * rank

    #         # 检查是否有足够的权重数据剩余
    #         if offset_A + num_elements_A > all_A_weights.numel() or offset_B + num_elements_B > all_B_weights.numel():
    #             logger.error(f"Ran out of weights while processing '{module_name}'. "
    #                          f"A_needed={num_elements_A}, A_left={all_A_weights.numel() - offset_A}. "
    #                          f"B_needed={num_elements_B}, B_left={all_B_weights.numel() - offset_B}.")
    #             break
                
    #         # 切分、重塑并存储 lora_A
    #         lora_A_flat = all_A_weights[offset_A : offset_A + num_elements_A]
    #         shape_A = (rank, in_features)
    #         unwrapped_weights[f"{module_name}.lora_A.weight"] = lora_A_flat.reshape(shape_A)
    #         offset_A += num_elements_A

    #         # 切分、重塑并存储 lora_B
    #         lora_B_flat = all_B_weights[offset_B : offset_B + num_elements_B]
    #         shape_B = (out_features, rank)
    #         unwrapped_weights[f"{module_name}.lora_B.weight"] = lora_B_flat.reshape(shape_B)
    #         offset_B += num_elements_B
            
    #         logger.info(
    #             f"Successfully carved and reshaped weights for: {module_name} | "
    #             f"A: {list(shape_A)}, B: {list(shape_B)}"
    #         )

    #     # 4. 最终健全性检查
    #     if offset_A != all_A_weights.numel():
    #         logger.warning(
    #             f"LoRA A weights buffer mismatch. Used {offset_A} elements, but buffer has {all_A_weights.numel()}. "
    #             f"This may be expected if not all target_modules exist in the model."
    #         )
    #     if offset_B != all_B_weights.numel():
    #          logger.warning(
    #             f"LoRA B weights buffer mismatch. Used {offset_B} elements, but buffer has {all_B_weights.numel()}. "
    #             f"This may be expected if not all target_modules exist in the model."
    #         )

    #     if not unwrapped_weights:
    #         logger.error("CRITICAL: Failed to unwrap any LoRA weights. No target modules were matched. "
    #                      "Please check model architecture and `target_modules` in LoRA config. "
    #                      "Enable log_level='debug' to see all checked module names.")

    #     return unwrapped_weights
    
   
    # def _convert_fsdp_parameters(self, lora_config: dict) -> dict:
    #     """
    #     最终诊断版本: 此函数不再尝试应用权重。
    #     它的唯一目的是检查vLLM模型内部的模块名称，并与你的LoRA配置进行比较，
    #     然后用一个清晰的错误消息来指导你如何修复配置。
    #     """
    #     logger.info("--- RUNNING IN DIAGNOSTIC MODE to identify LoRA target modules ---")
    #     print("lora_config: ",lora_config)
    #     for name, weight_param in self.lora_weight.items():
    #         print("for name, weight_param in self.lora_weight.items():",name)
    #     # 1. 从你的LoRA配置中获取目标模块
    #     target_modules_set = set(lora_config.get("target_modules", []))
    #     if not target_modules_set:
    #         raise ValueError("LoRA configuration must contain a non-empty 'target_modules' list.")

    #     # 2. 遍历vLLM模型，发现所有可能的LoRA目标层
    #     discovered_module_keys = set()
    #     discovered_full_module_names = []
    #     for module_name, module in self.model_runner.model.named_modules():
            
    #         # 记录完整的模块名和它的“键”（通常是最后一部分）
    #         discovered_full_module_names.append(module_name)
    #         module_key = module_name
    #         discovered_module_keys.add(module_key)

    #     # 3. 比较你的配置和我们发现的模块，然后生成一个决定性的错误报告
        
    #     matched_keys = discovered_module_keys.intersection(target_modules_set)

    #     # 无论是否匹配，我们都将抛出错误，以显示所有信息
    #     # 这确保你100%能看到正确的名称
        
    #     error_message = (
    #         "\n\n"
    #         "================================================================================\n"
    #         "DIAGNOSTIC REPORT: LoRA Target Module Analysis\n"
    #         "================================================================================\n"
    #         "This is a diagnostic message to help you correctly configure your LoRA training.\n"
    #         "The server will now stop. Please follow the instructions below.\n\n"
    #         "--- ANALYSIS ---\n"
    #         f"Your LoRA Config `target_modules`:\n"
    #         f"  {sorted(list(target_modules_set))}\n\n"
    #         f"Available Module Keys discovered in the vLLM model:\n"
    #         f"  {sorted(list(discovered_module_keys))}\n\n"
    #         f"Matched Keys: {sorted(list(matched_keys)) if matched_keys else 'NONE - THIS IS THE PROBLEM!'}\n\n"
    #         "--- ACTION REQUIRED ---\n"
    #         "1. COMPARE the two lists above ('Your LoRA Config' vs 'Available Module Keys').\n"
    #         "2. UPDATE the `target_modules` list in the `PeftConfig` of your TRAINING script\n"
    #         "   to use the correct names from the 'Available Module Keys' list.\n"
    #         "3. AFTER updating your training script, you will need to RESTART the training\n"
    #         "   from scratch, as the LoRA adapter structure has changed.\n\n"
    #         "--- EXAMPLE FULL MODULE NAMES (for reference) ---\n"
    #         + "\n".join([f"  - {name}" for name in discovered_full_module_names[:15]]) + "\n"
    #         "================================================================================\n"
    #     )

    #     raise ValueError(error_message)

    #     # The function will never reach here. It is designed to stop execution.
    #     return {}

    # def _convert_fsdp_parameters(self, lora_config: dict) -> dict:
    #     """
    #     终极解决方案 V6: "聚合-切分" (Aggregate-and-Carve) 策略。
    #     此方法放弃解析单个FSDP名称，而是将所有权重聚合起来，然后根据vLLM模型的结构顺序进行切分。
    #     这是应对FSDP权重大小和形状不确定性的最稳健方法。
    #     """
    #     unwrapped_weights = {}
    #     logger.info("Starting FSDP parameter conversion (V6) with Aggregate-and-Carve strategy...")

    #     # 1. 获取LoRA rank并准备vLLM的目标模块
    #     rank = lora_config.get("r")
    #     if not rank: raise ValueError("LoRA config 'r' (rank) is missing.")
        
    #     # 将用户配置的目标模块映射到vLLM的合并后模块名
    #     original_targets = set(lora_config.get("target_modules", []))
    #     vllm_targets = set()
    #     if any(t in original_targets for t in ["q_proj", "k_proj", "v_proj"]): vllm_targets.add("qkv_proj")
    #     if any(t in original_targets for t in ["gate_proj", "up_proj"]): vllm_targets.add("gate_up_proj")
    #     for target in original_targets:
    #         if target not in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]: vllm_targets.add(target)
    #     logger.info(f"Mapping training targets to vLLM targets: {sorted(list(vllm_targets))}")


    #     # 2. 聚合所有收到的FSDP权重，创建两个“权重池”
    #     flat_params_A, flat_params_B = {}, {}
    #     for name, weight_param in self.lora_weight.items():
    #         if ".lora_A." in name: flat_params_A[name] = weight_param
    #         elif ".lora_B." in name: flat_params_B[name] = weight_param
        
    #     # 必须按名称排序来保证顺序与FSDP的flat_param一致
    #     sorted_A_tensors = [flat_params_A[k] for k in sorted(flat_params_A.keys())]
    #     sorted_B_tensors = [flat_params_B[k] for k in sorted(flat_params_B.keys())]

    #     if not sorted_A_tensors or not sorted_B_tensors:
    #         logger.error("No LoRA A or B weights received from FSDP.")
    #         return {}

    #     all_A_weights = torch.cat([t.flatten() for t in sorted_A_tensors])
    #     all_B_weights = torch.cat([t.flatten() for t in sorted_B_tensors])
    #     logger.info(f"Aggregated LoRA A weights into a pool of size: {all_A_weights.numel()}")
    #     logger.info(f"Aggregated LoRA B weights into a pool of size: {all_B_weights.numel()}")

    #     offset_A, offset_B = 0, 0

    #     # 3. 严格按vLLM模型结构顺序，遍历并切分权重
    #     for module_name, module in self.model_runner.model.named_modules():
    #         module_key = module_name.split('.')[-1]

    #         if module_key not in vllm_targets:
    #             continue
            
    #         logger.info(f"Processing target module in vLLM: '{module_name}' (key: '{module_key}')")
            
    #         # 从 vLLM module 中获取真实的维度
    #         if hasattr(module, "weight"):
    #             # 对于 ColumnParallelLinear, RowParallelLinear 等
    #             out_features, in_features = module.weight.shape
    #             # vLLM的合并层可能需要特殊处理
    #             if module_key == "qkv_proj":
    #                 # qkv_proj 的 weight shape 是 [3 * head_dim * num_heads, hidden_size]
    #                 # 我们需要的是单个 Q/K/V 的维度，但这里直接用合并后的
    #                 # 注意: vLLM的LoRA实现内部会处理qkv的拆分
    #                 pass
    #             elif module_key == "gate_up_proj":
    #                 # gate_up_proj weight shape is [2 * intermediate, hidden_size]
    #                 out_features = out_features // 2 # 我们需要的是单个gate或up的维度
    #         elif hasattr(module, "embedding_dim"):
    #              # For Embedding
    #             in_features, out_features = module.num_embeddings, module.embedding_dim
    #         else:
    #             logger.warning(f"Cannot determine dimensions for module {module_name}, skipping.")
    #             continue

    #         # 计算此模块LoRA矩阵所需的元素数量
    #         num_elements_A = rank * in_features
    #         num_elements_B = out_features * rank
            
    #         # vLLM 合并层特殊处理
    #         if module_key == "qkv_proj":
    #             num_elements_B *= 3
    #         elif module_key == "gate_up_proj":
    #             num_elements_A *= 2
    #             num_elements_B *= 2

    #         # 检查权重池中是否有足够的权重
    #         if offset_A + num_elements_A > all_A_weights.numel() or offset_B + num_elements_B > all_B_weights.numel():
    #             logger.error(f"Ran out of weights while processing '{module_name}'. "
    #                          f"A needed={num_elements_A}, A left={all_A_weights.numel() - offset_A}. "
    #                          f"B needed={num_elements_B}, B left={all_B_weights.numel() - offset_B}.")
    #             break
            
    #         # 切分并重塑
    #         lora_A_flat = all_A_weights[offset_A : offset_A + num_elements_A]
    #         lora_B_flat = all_B_weights[offset_B : offset_B + num_elements_B]
            
    #         shape_A = (rank if module_key != "gate_up_proj" else rank * 2, in_features)
    #         shape_B = (out_features if module_key != "qkv_proj" else out_features * 3, rank if module_key != "gate_up_proj" else rank * 2)
    #         if module_key == "gate_up_proj":
    #             shape_B = (out_features*2, rank)

    #         # 修正 gate_up_proj B 的形状
    #         if module_key == "gate_up_proj":
    #             shape_A = (rank * 2, in_features)
    #             shape_B = (out_features * 2, rank)
    #         # 修正 qkv_proj B 的形状
    #         elif module_key == "qkv_proj":
    #             shape_A = (rank, in_features)
    #             shape_B = (out_features * 3, rank)
    #         else:
    #             shape_A = (rank, in_features)
    #             shape_B = (out_features, rank)

    #         unwrapped_weights[f"{module_name}.lora_A.weight"] = lora_A_flat.reshape(shape_A)
    #         unwrapped_weights[f"{module_name}.lora_B.weight"] = lora_B_flat.reshape(shape_B)

    #         offset_A += num_elements_A
    #         offset_B += num_elements_B
    #         logger.info(f"Successfully carved weights for: {module_name} | A: {list(shape_A)}, B: {list(shape_B)}")

    #     # 4. 最终健全性检查
    #     logger.info("--- Running Final Sanity Checks ---")
    #     if abs(offset_A - all_A_weights.numel()) > 1 or abs(offset_B - all_B_weights.numel()) > 1: # Allow for off-by-one errors
    #         logger.warning(
    #             f"LoRA weights buffer mismatch. A used/total: {offset_A}/{all_A_weights.numel()}. "
    #             f"B used/total: {offset_B}/{all_B_weights.numel()}. "
    #             "This can happen if target_modules in training and vLLM do not perfectly align."
    #         )
    #     else:
    #          logger.info("+++ Sanity Check PASSED: All aggregated weights were consumed. +++")
    #     if not unwrapped_weights:
    #         raise RuntimeError("CRITICAL: Failed to unwrap any LoRA weights. No target modules were matched.")
        
    #     return unwrapped_weights

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
     
    # def apply_lora(self, lora_config: dict) -> int:
    #     """使用收集到的LoRA权重创建并注册LoRA适配器"""
    #     if not self.lora_weight:
    #         raise ValueError("No LoRA weights collected. Call apply_lora first.")
    #     # for name, weight_param in self.lora_weight.items():
    #     #     print(name)
    #     # 创建唯一的LoRA ID
    #     self.lora_id += 1
    #     # 处理FSDP包装的参数
    #     processed_weights = self._convert_fsdp_parameters()
    #     # 创建LoRA请求对象
    #     lora_request = PatchedLoRARequest(
    #         lora_name=f"lora_{self.lora_id}",
    #         lora_int_id=self.lora_id,
    #         lora_tensors=processed_weights,#processed_weights self.lora_weight
    #         lora_config=lora_config,
    #     )
        
    #     # 注册LoRA适配器
    #     success = self.add_lora(lora_request)
    #     if not success:
    #         raise RuntimeError(f"Failed to add LoRA adapter ID {self.lora_id}")
        
    #     # 清空权重缓存
    #     self.lora_weight = {}
    #     return self.lora_id

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

    # # 修改 apply_lora 来调用新的转换函数
    # def apply_lora(self, lora_config: dict) -> int:
    #     if not self.lora_weight: raise ValueError("No LoRA weights collected.")
    #     self.lora_id += 1
        
    #     # 调用新的、简化的转换与合并函数
    #     processed_weights = self._convert_and_merge_lora_weights()

    #     # 更新 lora_config 以匹配 vLLM 的合并层
    #     original_targets = set(lora_config.get("target_modules", []))
    #     vllm_targets = set()
    #     if any(t in original_targets for t in ["q_proj", "k_proj", "v_proj"]): vllm_targets.add("qkv_proj")
    #     if any(t in original_targets for t in ["gate_proj", "up_proj"]): vllm_targets.add("gate_up_proj")
    #     for target in original_targets:
    #         if target not in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]: vllm_targets.add(target)

    #     updated_lora_config = lora_config.copy()
    #     updated_lora_config['target_modules'] = sorted(list(vllm_targets))
        
    #     lora_request = PatchedLoRARequest(
    #         lora_name=f"lora_{self.lora_id}", lora_int_id=self.lora_id,
    #         lora_tensors=processed_weights, lora_config=updated_lora_config,
    #     )
        
    #     self.lora_requests = lora_request

    #     logger.info(f"Successfully applied LoRA adapter with ID: {self.lora_id}")
    #     return self.lora_id

     
    def apply_lora(self, lora_config: dict) -> int:
        """使用收集到的LoRA权重创建并注册LoRA适配器"""
        from vllm.lora.request import LoRARequest
        # for name, weight_param in self.lora_weight.items():
        #     print(name)
        # 创建唯一的LoRA ID
        self.lora_id += 1
        self.lora_config = lora_config
        # 处理FSDP包装的参数
        #processed_weights = self._convert_fsdp_parameters()
        # 创建LoRA请求对象
        lora_request = LoRARequest(
            lora_name=f"lora_{self.lora_id}",
            lora_int_id=self.lora_id,
            lora_tensors=self.lora_weight,#processed_weights self.lora_weight
            lora_config=lora_config,
        )
        self.lora_requests = lora_request
        self.add_lora(self.lora_requests)
        return self.lora_id
    
    def get_lora_request(self):
        """一个简单的 RPC 目标，返回缓存的 LoRA 请求对象。"""
        from vllm.lora.request import LoRARequest
        # 处理FSDP包装的参数
        # 创建LoRA请求对象
        lora_request = LoRARequest(
            lora_name=f"lora_{self.lora_id}",
            lora_int_id=self.lora_id,
            lora_tensors=self.lora_weight, ##processed_weights self.lora_weight
            lora_config=self.lora_config,
        )
        return lora_request

    def get_lora_id(self):
        """只返回 LoRA 的整数 ID，避免序列化复杂对象。"""
        if self.lora_requests:
            return self.lora_requests.lora_int_id
        return 0 # 或者 -1，表示没有激活的 LoRA
    # # 这个方法也在 worker 内部执行
    def generate_with_lora(self, prompts: list[str], sampling_params: SamplingParams):
        return self.model_executor.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=self.lora_requests
        )
    
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
        enable_lora=True,
        max_lora_rank=512,
        max_cpu_loras=1,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
        trust_remote_code=script_args.trust_remote_code,
    )
    print(f"--- [DP RANK {data_parallel_rank}] DIAGNOSTICS ---")
    print(f"llm object type: {type(llm)}")
    print(f"llm_engine object type: {type(llm.llm_engine.engine_core)}")
    print(f"Attributes of llm_engine: {dir(llm.llm_engine.engine_core)}")
    print(f"--- END DIAGNOSTICS ---")
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
        # ==================== 添加新的命令分支 ====================
        
        # ==========================================================
        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            # if method_name == "generate":
            #     worker_extension = llm.llm_engine.engine_core.model_executor.workers[0].worker_extension
            #     kwargs["lora_request"] = worker_extension.lora_requests
             # ================== 智能拦截与注入 ==================
            # 如果是 generate 命令，并且没有显式提供 lora_request
            if method_name == "generate" and "lora_request" not in kwargs:
                # 1. 从 worker 进程的子进程（TP workers）中获取 LoRA 请求。
                #    我们必须使用 RPC，因为 LoRA 状态存在于子进程中。
                lora_results_list = llm.collective_rpc(method="get_lora_request")

                # 2. 从返回的列表中提取单个 LoRARequest 对象。
                single_lora_request = None
                if lora_results_list and len(lora_results_list) > 0:
                    single_lora_request = lora_results_list[0]
                
                # 3. 将获取到的 LoRA 请求对象注入到 kwargs 中。
                #    这样，接下来的 llm.generate 调用就会带上这个参数。
                if single_lora_request:
                    kwargs["lora_request"] = single_lora_request
            # ========================================================
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


            # # # ======================= 核心修改在这里 =======================
            # # 1. 先通过 RPC 从 worker 获取 lora_request
            # get_lora_command = {
            #     "type": "call",
            #     "method": "collective_rpc",
            #     "args": ("get_lora_request", None, (), {})
            # }
            # connection.send(get_lora_command)
            # lora_request_results = await asyncio.to_thread(connection.recv)
            # # ======================= 核心修改在这里 =======================
            # single_lora_request = lora_request_results[0]
            # lora_request_results 的结构是 [[...]]，所以要取 [0]
            # if lora_request_results and lora_request_results[0]:
            #     # 从序列化后的列表中提取数据
            #     lora_data_list = lora_request_results[0]
                
                # # 手动反序列化，重新创建 LoRARequest 对象
                # # 注意：这里我们使用 PatchedLoRARequest，因为它支持 lora_tensors
                # # 这里的索引对应 LoRARequest 的 __getstate__ 返回的元组结构
                # single_lora_request = PatchedLoRARequest(
                #     lora_name=lora_data_list[0],
                #     lora_int_id=lora_data_list[1],
                #     # 第三个是 lora_local_path，我们传空字符串
                #     # 第四个是 lora_tensors，我们传空字典，因为权重已在 worker 端
                #     # 第五个是 lora_config
                #     lora_local_path=lora_data_list[2],
                #     lora_tensors={}, # 关键：这里传空字典
                #     lora_config=lora_data_list[4], 
                # )

            # ==========================================================

            # # # 2. 如果获取到了 LoRA 请求，就根据 prompts 的数量复制它
            # final_lora_request = None
            # if single_lora_request is not None:
            #     # 创建一个长度与 prompts 相同的列表，每个元素都是同一个 LoRA 请求对象
            #     final_lora_request = [single_lora_request] * len(prompts)
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
