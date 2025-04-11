%%writefile model.py

import torch
import torch.nn as nn
from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Authenticate with your token
login(token="hf_token")  # Replace with actual token

class HybridDP_LoRA(nn.Module):
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        lora_rank: int = 8,
        use_4bit: bool = True
    ):
        super().__init__()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if use_4bit else None
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_auth_token=True  # Critical for gated models
        )
        
        self.base_model.gradient_checkpointing_enable()
        
        self.model = get_peft_model(
            self.base_model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM"
            )
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable} ({100*trainable/total:.2f}%)")
