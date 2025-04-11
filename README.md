---

## **Project Overview**

HybridDP-LoRA is a novel framework for privacy-preserving federated fine-tuning of large language models (LLMs). It addresses the trade-off between differential privacy guarantees and personalization, achieving state-of-the-art results through innovations like dynamic rank adaptation, sparsity-weighted aggregation, and gradient checkpointing.

---

## **Repository Structure**

| **File/Folder** | **Description** |
| :-- | :-- |
| `model.py` | Implements the HybridDP-LoRA model with 4-bit quantization and LoRA integration. |
| `train.py` | Contains the training loop for single-client and federated learning workflows. |
| `fl_client.py` | Defines the Flower client logic for federated learning with DP integration. |
| `dp_engine.py` | Implements the Adaptive Differential Privacy Engine for noise scaling. |
| `evaluation.py` | Comprehensive evaluation suite for privacy, utility, and efficiency metrics. |
| `utils.py` | Utility functions for data preprocessing, logging, and dataset loading. |
| `requirements.txt` | Lists all dependencies required to run the project (e.g., PyTorch, Flower). |


---

## **Key Features**

1. **Two-Stage Hybrid Privacy Architecture**:
    - Global Stage: Trains LoRA matrices with DP guarantees.
    - Local Stage: Personalizes LoRA-B matrices without DP constraints.
2. **Dynamic Rank Adaptation**:
    - Automatically adjusts LoRA ranks (1–16) based on client heterogeneity.
3. **Communication Efficiency**:
    - 3,300× compression via sparsity-weighted aggregation and 8-bit quantization.
4. **Evaluation Suite**:
    - Validates privacy-utility trade-offs, communication costs, and personalization gains.

---

## **How to Run**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```


### **2. Train a Single Client**

```bash
python train.py --mode single-client --epochs 10
```


### **3. Run Federated Learning Simulation**

```bash
python train.py --mode federated --num_clients 5 --rounds 3
```


### **4. Evaluate Model**

```bash
python evaluation.py --model_path &lt;path_to_model&gt; --dataset &lt;path_to_dataset&gt;
```

---
