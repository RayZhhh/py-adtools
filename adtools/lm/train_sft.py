"""
Copyright (c) 2025 Rui Zhang <rzhang.cs@gmail.com>

NOTICE: This code is under MIT license. This code is intended for academic/research purposes only.
Commercial use of this software or its derivatives requires prior written permission.
"""

import os
import argparse
import sys
from pathlib import Path

from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_file', required=True)
    p.add_argument('--model_name_or_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--adapter_path', type=str, default=None)
    p.add_argument('--per_device_batch_size', type=int, default=2)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--num_train_epochs', type=int, default=5)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--lora_target_modules', type=str,
                   default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--save_merged_model', action='store_true')
    p.add_argument('--flash_attn', action='store_true')
    p.add_argument('--deepspeed', type=str, default=None)
    return p.parse_args()


def longest_seq_len(dataset, tok):
    return max(
        len(tok(example['prompt'] + example['completion']).input_ids)
        for example in dataset
    )


def main() -> None:
    # Suppress output if current process is not rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        sys.stdout = open(os.devnull, 'w')

    args = parse_args()
    dataset = load_dataset('json', data_files=args.train_file, split='train')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Initialize an SFT config before loading the model
    # This enables ZeRO stage-3 parameter partitioning
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=longest_seq_len(dataset, tokenizer),
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=torch.cuda.is_bf16_supported(),
        deepspeed=args.deepspeed,
        packing=True if args.flash_attn else False,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype='auto',
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if args.flash_attn else None,
    )

    # Wrap it with PeftModel
    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=True
        )
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=args.lora_target_modules.split(','),
        )
        model = get_peft_model(model, lora_cfg)

    # Create a trainer and train
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save lora adapter
    peft_model = trainer.model
    os.makedirs(Path(args.output_dir) / 'last_lora_weights', exist_ok=True)
    peft_model.save_pretrained(Path(args.output_dir) / 'last_lora_weights')

    # Save merged model
    if args.save_merged_model:
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir)
    else:  # Save lora adapter
        peft_model.save_pretrained(args.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Destroy process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
