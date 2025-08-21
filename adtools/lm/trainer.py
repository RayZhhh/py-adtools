"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import subprocess
import os
import sys
from os import PathLike
from pathlib import Path
from typing import Optional, List, Literal, Dict


def _print_cmd_list(trainer_name, cmd_list, gpus):
    print('\n' + '=' * 80)
    print(f'[{trainer_name}] Launching Trainer on GPU:{gpus}')
    print('=' * 80)
    cmd = cmd_list[0] + ' \\\n'
    for c in cmd_list[1:]:
        cmd += '    ' + str(c) + ' \\\n'
    print(cmd.strip())
    print('=' * 80 + '\n', flush=True)


def launch_torchrun_sft(
        model_name_or_path: str,
        train_file: str,
        device_ids: Optional[int | List[int]],
        output_dir: str,
        adapter_path: Optional[str] = None,
        torchrun_master_port: int = 39500,
        epoch: int = 2,
        per_device_batch_size: int = 1,
        grad_accumulate_steps: int = 2,
        learning_rate: float = 5e-5,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        flash_attn: bool = False,
        deepspeed_config_path: Optional[str] = None,
        lora_target_module: str = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        save_merged_model: bool = False,
        env_variables: Dict[str, str] = None
) -> bool:
    """Run torchrun for SFT training with the provided script and environment settings.
    Returns `True` if training success, otherwise returns `False`.
    """
    script_path = Path(__file__).parent / 'train_sft.py'

    # Set CUDA_VISIBLE_DEVICES environment variable
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    device_ids = [str(i) for i in device_ids]
    cuda_visible_devices_str = ','.join(device_ids)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices_str

    # Set NCCL environment variable
    env['NCCL_P2P_DISABLE'] = '1'

    # Set other environment variables
    if env_variables is not None:
        for k in env_variables.keys():
            # export 'KEY'='VALUE'
            env[k] = env_variables[k]

    # Construct the torchrun command
    executable_path = sys.executable
    command = [
        str(executable_path), '-m', 'torch.distributed.run',
        '--nproc_per_node', str(len(device_ids)),
        '--nnodes', '1',
        '--node_rank', '0',
        '--master_port', str(torchrun_master_port),
        str(script_path),
        '--train_file', train_file,
        '--model_name_or_path', model_name_or_path,
        '--output_dir', output_dir,
        '--per_device_batch_size', str(per_device_batch_size),
        '--gradient_accumulation_steps', str(grad_accumulate_steps),
        '--num_train_epochs', str(epoch),
        '--learning_rate', str(learning_rate),
        '--lora_rank', str(lora_rank),
        '--lora_alpha', str(lora_alpha),
        '--lora_target_module', str(lora_target_module),
    ]

    if adapter_path:
        command.extend(['--adapter_path', adapter_path])
    if deepspeed_config_path is not None:
        command.extend(['--deepspeed', str(deepspeed_config_path)])
    if flash_attn:
        command.append('--flash_attn')
    if save_merged_model:
        command.append('--save_merged_model')

    # Run the command and wait for it to finish
    _print_cmd_list('SFTTrainer', cmd_list=command, gpus=cuda_visible_devices_str)
    result = subprocess.run(command, env=env, check=True)

    # Check if the command was successful
    if result.returncode == 0:
        print('[SFTTrainer] SFT torchrun finished successfully.')
        return True
    else:
        print(f'[SFTTrainer] SFT torchrun failed with return code {result.returncode}.')
        return False


def launch_accelerate_sft(
        model_name_or_path: str,
        train_file: str,
        device_ids: Optional[int | List[int]],
        output_dir: str,
        adapter_path: Optional[str] = None,
        main_process_port: int = 39500,
        epoch: int = 2,
        per_device_batch_size: int = 1,
        grad_accumulate_steps: int = 2,
        learning_rate: float = 5e-5,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        flash_attn: bool = False,
        deepspeed_config_path: Optional[str] = None,
        lora_target_module: str = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
        save_merged_model: bool = False,
        env_variables: Dict[str, str] = None
) -> bool:
    """Run torchrun for SFT training with the provided script and environment settings.
    Returns `True` if training success, otherwise returns `False`.
    """
    script_path = Path(__file__).parent / 'train_sft.py'

    # Set CUDA_VISIBLE_DEVICES environment variable
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    device_ids = [str(i) for i in device_ids]
    cuda_visible_devices_str = ','.join(device_ids)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices_str

    # Set NCCL environment variable
    env['NCCL_P2P_DISABLE'] = '1'

    # Set other environment variables
    if env_variables is not None:
        for k in env_variables.keys():
            # export 'KEY'='VALUE'
            env[k] = env_variables[k]

    # Construct the torchrun command
    executable_path = sys.executable
    command = [
        str(executable_path), '-m', 'accelerate.commands.launch',
        '--num_processes', str(len(device_ids)),
        '--num_machines', '1',
        '--main_process_port', str(main_process_port),
        str(script_path),
        '--train_file', train_file,
        '--model_name_or_path', model_name_or_path,
        '--output_dir', output_dir,
        '--per_device_batch_size', str(per_device_batch_size),
        '--gradient_accumulation_steps', str(grad_accumulate_steps),
        '--num_train_epochs', str(epoch),
        '--learning_rate', str(learning_rate),
        '--lora_rank', str(lora_rank),
        '--lora_alpha', str(lora_alpha),
        '--lora_target_module', str(lora_target_module),
    ]

    if adapter_path:
        command.extend(['--adapter_path', adapter_path])
    if deepspeed_config_path is not None:
        command.extend(['--deepspeed', str(deepspeed_config_path)])
    if flash_attn:
        command.append('--flash_attn')
    if save_merged_model:
        command.append('--save_merged_model')

    # Run the command and wait for it to finish
    _print_cmd_list('SFTTrainer', cmd_list=command, gpus=cuda_visible_devices_str)
    result = subprocess.run(command, env=env, check=True)

    # Check if the command was successful
    if result.returncode == 0:
        print('[SFTTrainer] SFT accelerate launch finished successfully.')
        return True
    else:
        print(f'[SFTTrainer] SFT accelerate launch failed with return code {result.returncode}.')
        return False
