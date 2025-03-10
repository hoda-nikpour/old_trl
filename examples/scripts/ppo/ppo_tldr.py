# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import shutil
import torch
import os

# Ensure world_size is set properly
os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())  # Automatically set world_size based on GPUs
torch.set_default_dtype(torch.bfloat16)

# Check if it's correctly set
print(f"WORLD_SIZE is set to: {os.environ['WORLD_SIZE']}")

import torch
from accelerate import PartialState  # Make sure you have accelerate>=0.17.0
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from custom_agent.agent_dataset import AgentDataset
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
Example usage (single-GPU):

python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100

Multi-GPU usage with DeepSpeed Zero2:

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100
"""


if __name__ == "__main__":
    # Parse all our arguments
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    print("---------------")
    print(script_args)
    print("---------------")
    # Remove output_dir if it already exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ########################################################################
    # 1. Load and configure models & tokenizer
    ########################################################################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] 
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # If the tokenizer doesn't have a chat_template, fall back to the simple one
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Value model & reward model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    # Policy model
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    # If we have a LoRA or PEFT config, the reference policy will be the base model
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ########################################################################
    # 2. Load and preprocess dataset
    ########################################################################
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    #Hoda:
    # load_dataset(script_args.dataset_name,name=script_args.dataset_config)

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = (
        dataset[script_args.dataset_test_split] 
        if training_args.eval_strategy != "no" 
        else None
    )


    with PartialState().local_main_process_first():
        train_dataset = AgentDataset(script_args.dataset_name, tokenizer)
        # prepare_dataset(train_dataset, tokenizer)
        # if eval_dataset is not None:
        #     eval_dataset = prepare_dataset(eval_dataset, tokenizer)

        # Filter out sequences that would exceed 512 in length
        # train_dataset = train_dataset.filter(
        #     lambda x: x["lengths"] <= 512, 
        #     num_proc=training_args.dataset_num_proc,
        # )
        # if eval_dataset is not None:
        #     eval_dataset = eval_dataset.filter(
        #         lambda x: x["lengths"] <= 512, 
        #         num_proc=training_args.dataset_num_proc,
        #     )

    # Ensure that the last token is not already the EOS token
    assert (
        train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    ), "The last token should not be an EOS token"

    ########################################################################
    # 3. Create and run the PPOTrainer
    ########################################################################
    trainer = PPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        paper_db=training_args.paper_db,
        paper_id=training_args.paper_id,
    )

    trainer.train()

    # trainer = FixZero3CheckpointPPOTrainer(
    #     config=training_args,
    #     processing_class=tokenizer,
    #     policy=policy,
    #     ref_policy=ref_policy,
    #     value_model=value_model,
    #     train_dataset=train_dataset,
    #     paper_db=training_args.paper_db,
    #     paper_id=training_args.paper_id,
    # )
    # trainer.train()

    # # Save final model to output_dir
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # # Optionally, generate some sample completions for verification
    # trainer.generate_completions()

# # Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import shutil

# from accelerate import PartialState
# from transformers import (
#     AutoModelForCausalLM,
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     HfArgumentParser,
#     Trainer
# )

# from trl import ModelConfig, PPOConfig, PPOTrainer, ScriptArguments
# from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
# from custom_agent.agent_dataset import AgentDataset
# from typing import Optional
# import torch.nn as nn

# class FixZero3CheckpointPPOTrainer(PPOTrainer):

#     def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
#         backup_model = self.model
#         self.model = self.model.policy  # save only the policy

#         Trainer.save_model(self, output_dir, _internal_call)

#         self.model = backup_model

#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         if self.is_deepspeed_enabled:
#             state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
#                           if name.startswith('policy.')}

#         super()._save(output_dir, state_dict)

# if __name__ == "__main__":
#     parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
#     script_args, training_args, model_config = parser.parse_args_into_dataclasses()
#     shutil.rmtree(training_args.output_dir, ignore_errors=True)

#     # tokenizer and dataset
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_config.model_name_or_path,
#         padding_side="left",
#         trust_remote_code=model_config.trust_remote_code
#     )
#     train_dataset = AgentDataset(script_args.dataset_name, tokenizer)
#     assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

#     # models
#     value_model = AutoModelForSequenceClassification.from_pretrained(
#         training_args.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
#     )
#     for m in value_model.score.modules():
#         if isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, mean=0, std=0.01)
#     ref_policy = AutoModelForCausalLM.from_pretrained(
#         training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
#     )
#     policy = AutoModelForCausalLM.from_pretrained(
#         training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
#     )

#     trainer = FixZero3CheckpointPPOTrainer(
#         config=training_args,
#         processing_class=tokenizer,
#         policy=policy,
#         ref_policy=ref_policy,
#         value_model=value_model,
#         train_dataset=train_dataset,
#         paper_db=training_args.paper_db,
#         paper_id=training_args.paper_id,
#     )
#     trainer.train()

#     # Save and push to hub
#     trainer.save_model(training_args.output_dir)
