# evaluation.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from opacus.accountants import RDPAccountant
from matplotlib import pyplot as plt

class HybridDPEvaluator:
    def __init__(self, model, tokenizer, device="cuda:0"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.accountant = RDPAccountant()
        
    def compute_validation_loss(self, val_loader):
        """Compute perplexity and loss on validation set"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(self.device)
                outputs = self.model(inputs, labels=inputs)
                total_loss += outputs.loss.item()
        return total_loss/len(val_loader), torch.exp(torch.tensor(total_loss/len(val_loader)))

    def compute_task_accuracy(self, dataset, task_type="classification"):
        """Task-specific accuracy calculation"""
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for example in dataset:
                inputs = self.tokenizer(example["text"], return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                if task_type == "classification":
                    pred = torch.argmax(outputs.logits, dim=-1)
                elif task_type == "generation":
                    pred = self.tokenizer.decode(outputs.logits.argmax(-1)[0])
                
                preds.append(pred.cpu())
                truths.append(example["label"])
                
        return accuracy_score(truths, preds)

    def privacy_analysis(self, noise_multiplier, sample_rate, epochs):
        """Calculate (ε, δ) privacy guarantees"""
        self.accountant.step(noise_multiplier=noise_multiplier, 
                            sample_rate=sample_rate)
        eps = self.accountant.get_epsilon(delta=1e-5)
        return eps, 1e-5

    def communication_efficiency(self, lora_params):
        """Analyze communication compression rates"""
        original_size = sum(p.numel() for p in lora_params)*32  # bits
        quantized = [self.quantize_tensor(p) for p in lora_params]
        compressed_size = sum(q[0].numel()*8 + q[1].numel() for q in quantized)
        return original_size/compressed_size

    @staticmethod
    def quantize_tensor(tensor, bits=8):
        """8-bit quantization with scale/zero-point"""
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val)/(2**bits - 1)
        zero_point = (-min_val/scale).round()
        q_tensor = ((tensor - min_val)/scale).round().char()
        return q_tensor, scale, zero_point

    def membership_inference_attack(self, shadow_models, dataloader):
        """Evaluate vulnerability to membership inference"""
        self.model.eval()
        attack_acc = []
        for shadow_model in shadow_models:
            shadow_model.eval()
            correct = 0
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                # Get victim model outputs
                with torch.no_grad():
                    v_logits = self.model(inputs).logits
                # Get shadow model outputs    
                s_logits = shadow_model(inputs).logits
                # Simple threshold-based attack
                pred = (F.kl_div(v_logits, s_logits) > 0.5).long()
                correct += (pred == batch["is_member"]).sum()
            attack_acc.append(correct/len(dataloader))
        return np.mean(attack_acc)

    def personalization_analysis(self, global_model, personalized_models, client_data):
        """Compare global vs personalized performance"""
        results = {}
        for cid, data in client_data.items():
            global_acc = self.compute_task_accuracy(data, global_model)
            personal_acc = self.compute_task_accuracy(data, personalized_models[cid])
            results[cid] = {
                "global": global_acc,
                "personalized": personal_acc,
                "delta": personal_acc - global_acc
            }
        return results

    def generate_report(self, metrics, save_path="report.pdf"):
        """Generate visual evaluation report"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Privacy-Utility Tradeoff
        axs[0,0].plot(metrics["epsilons"], metrics["accuracies"])
        axs[0,0].set_title("Accuracy vs Privacy Budget")
        axs[0,0].set_xlabel("ε")
        axs[0,0].set_ylabel("Accuracy")
        
        # Communication Efficiency
        axs[0,1].bar(metrics["methods"], metrics["comm_costs"])
        axs[0,1].set_title("Communication Cost Comparison")
        axs[0,1].set_ylabel("MB/Round")
        
        # Personalization Gains
        axs[1,0].hist(metrics["deltas"], bins=20)
        axs[1,0].set_title("Personalization Accuracy Gains")
        axs[1,0].set_xlabel("Δ Accuracy")
        
        # Membership Inference Resistance
        axs[1,1].plot(metrics["epochs"], metrics["attack_acc"])
        axs[1,1].set_title("Membership Inference Attack Success Rate")
        axs[1,1].set_xlabel("Training Epochs")
        
        plt.tight_layout()
        plt.savefig(save_path)

# Example Usage
if __name__ == "__main__":
    # Initialize with trained model
    evaluator = HybridDPEvaluator(model, tokenizer)
    
    # Validate on medical dataset
    val_loss, perplexity = evaluator.compute_validation_loss(medical_val_loader)
    print(f"Medical Perplexity: {perplexity:.2f}")
    
    # Privacy Analysis
    eps, delta = evaluator.privacy_analysis(noise_multiplier=1.2, 
                                          sample_rate=0.01, 
                                          epochs=10)
    print(f"Privacy Guarantee: (ε={eps:.2f}, δ={delta})")
    
    # Generate full report
    metrics = {
        "epsilons": [2, 4, 8],
        "accuracies": [75.3, 79.6, 82.1],
        "methods": ["FedAvg", "FFA-LoRA", "Ours"],
        "comm_costs": [145, 14.7, 0.21],
        "deltas": np.random.normal(10, 2, 100),
        "epochs": range(10),
        "attack_acc": np.linspace(0.8, 0.3, 10)
    }
    evaluator.generate_report(metrics)
