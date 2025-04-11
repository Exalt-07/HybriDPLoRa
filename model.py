# Define HybridDP-LoRA model
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class HybridDP_LoRA(nn.Module):
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        lora_rank: int = 8,
        use_4bit: bool = True
    ):
        super().__init__()
        
        # Setup quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if use_4bit else None
        
        # HF authentication (if needed)
        try:
            from huggingface_hub import login
            login(token=os.environ.get("hf_token", ""))
        except:
            print("No authentication token found. Proceeding without login.")
        
        # Load model with explicit device control
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Auto-distribute on available GPU(s)
            torch_dtype=torch.bfloat16
        )
        
        # Memory optimizations
        self.base_model.gradient_checkpointing_enable()
        
        # Apply LoRA
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
        
        # Cache property for convenience
        self.device = next(self.parameters()).device

    def forward(self, input_ids, labels=None):
        """Forward pass with explicit device handling"""
        return self.model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
    
    def freeze_lora_A(self):
        """Freeze LoRA A matrices for local personalization stage"""
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                param.requires_grad_(False)
    
    def print_trainable_parameters(self):
        """Display parameter statistics"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable} ({100*trainable/total:.2f}%)")
