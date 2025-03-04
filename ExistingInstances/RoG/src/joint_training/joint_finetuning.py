import datasets
from peft import AutoPeftModelForCausalLM, LoraConfig
from align_kg.data_loader import load_multiple_datasets, load_new_tokens
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.trainer_utils import get_last_checkpoint
import logging
from utils import *
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from typing import Optional
from dataclasses import dataclass, field
import sys
import os

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


datasets.disable_progress_bar()

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']
             ) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """
SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'


@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(
        metadata={"help": "Path to the training data."}
    )
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    rel_dict_path: list[str] = field(
        default=None, metadata={"help": "Path to the relation dictionary."}
    )
    add_rel_token: Optional[bool] = field(
        default=False, metadata={"help": "Wether to add relation token or not"}
    )
    prompt_path: str = field(
        default="prompts/llama2.txt",
        metadata={"help": "Path to the prompt template"},
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"}
    )


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    ddp_find_unused_parameters: bool = field(default=False)
    packing: bool = field(default=False)
    dataset_num_proc: int = field(default=N_CPUS)
    dataset_batch_size: int = field(default=1024)
    num_of_sequences: int = field(default=1024)
    chars_per_token: float = field(default=3.6)
    dataset_kwargs: dict = field(default=None)


def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_auth_token=True,
    )

    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # Add new tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<PAD>'
    new_tokens = [SEP, BOP, EOP]
    if script_args.add_rel_token:
        new_tokens = load_new_tokens(new_tokens, script_args.rel_dict_path)
    smart_tokenizer_and_embedding_resize(
        new_tokens, special_tokens_dict, tokenizer, model)

    # tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load datasets
    train_dataset = load_multiple_datasets(
        script_args.data_path_list, shuffle=True)

    # Prepare instruct tuning
    response_template = "[/INST]"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=training_args.model_max_length,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        packing=False,
        data_collator=data_collator,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({
                    training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {
                    last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    if script_args.use_peft:
        trainer.model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        if script_args.save_merged:
            del model
            torch.cuda.empty_cache()
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir, device_map="auto", torch_dtype=torch.bfloat16
            )
            model = model.merge_and_unload()
            model.eval()
            model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
