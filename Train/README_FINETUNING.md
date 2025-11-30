# Fine-Tuning GPT-OSS-20B Abliterated Model for Creative Writing

This guide explains how to fine-tune your abliterated GPT-OSS-20B model using Axolotl with LoRA for creative writing enhancement.

## Overview

- **Model**: GPT-OSS-20B (abliterated version)
- **Task**: Causal Language Modeling (CLM) for creative writing
- **Method**: LoRA (Low-Rank Adaptation) - parameter-efficient fine-tuning
- **Framework**: Axolotl
- **Dataset Format**: JSONL with `{"text":"STORY"}` format

## Prerequisites

1. **Hardware Requirements**:
   - RTX 6000 Pro (48GB VRAM) or RTX 3090 (24GB VRAM)
   - Multiple GPUs recommended for faster training
   - CUDA-compatible GPU drivers

2. **Software Requirements**:
   - Python 3.8+
   - CUDA toolkit
   - PyTorch with CUDA support
   - Axolotl framework

## Installation

### 1. Install Axolotl

```bash
# Clone Axolotl repository
git clone https://github.com/axolotl-ai/axolotl.git
cd axolotl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Axolotl
pip install -e .

# IMPORTANT: Update transformers to latest version (required for GPT-OSS support)
pip install --upgrade transformers
# Or install from source for absolute latest GPT-OSS support:
# pip install git+https://github.com/huggingface/transformers.git
```

### 2. Verify Installation

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Verify Axolotl installation
axolotl --version

# Verify transformers version (should be recent for GPT-OSS support)
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Dataset Preparation

Your dataset (`Datasets/dataset_20251128_154142.jsonl`) is already in the correct format. Split it into training and validation sets:

```bash
# From the repository root
python Train/split_dataset.py Datasets/dataset_20251128_154142.jsonl --val-ratio 0.1

# Or from Train directory
cd Train
python split_dataset.py ../Datasets/dataset_20251128_154142.jsonl --val-ratio 0.1 --output-dir ../Datasets
```

This will create:
- `Datasets/dataset_20251128_154142_train.jsonl` (90% of data)
- `Datasets/dataset_20251128_154142_val.jsonl` (10% of data)

## Configuration

### Model Path

Update the `base_model` path in `finetune_gpt_oss_20b.yaml` if your abliterated model is in a different location:

```yaml
base_model: /path/to/your/abliterated/model
```

The default path is set to `/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned` based on your `gpt-oss-20b.yml` configuration.

### Dataset Configuration

The config file specifies both training and validation datasets separately. If Axolotl doesn't recognize this format, you can:

1. **Option A (Recommended)**: Use the pre-split datasets as configured
2. **Option B**: Combine datasets back and let Axolotl split:
   ```bash
   cat Datasets/dataset_20251128_154142_train.jsonl Datasets/dataset_20251128_154142_val.jsonl > Datasets/dataset_20251128_154142_combined.jsonl
   ```
   Then update the config to use the combined file with `val_set_size: 0.1`

### Multi-GPU Configuration

Choose the appropriate configuration based on your hardware:

#### For RTX 6000 Pros (48GB VRAM each)
- Use **FSDP** (already configured in `finetune_gpt_oss_20b.yaml`)
- No changes needed

#### For RTX 3090s (24GB VRAM each)
- If memory constrained, use **DeepSpeed ZeRO Stage 2**:
  1. Comment out the `fsdp:` section in `finetune_gpt_oss_20b.yaml`
  2. Uncomment the `deepspeed:` line
  3. Ensure `deepspeed_config.json` is in the same directory

### Hyperparameter Tuning

Key parameters you may want to adjust:

- **`lora_r`**: LoRA rank (default: 8, try 16 for potentially better results)
- **`learning_rate`**: Learning rate (default: 3e-4, try 2e-4 to 5e-4)
- **`num_epochs`**: Number of training epochs (default: 3)
- **`per_device_train_batch_size`**: Batch size per GPU (default: 1, increase if you have VRAM)
- **`gradient_accumulation_steps`**: Effective batch size multiplier (default: 8)
- **`max_seq_length`**: Maximum sequence length (default: 2048, increase if model supports longer context)

## Training

### Start Training

**Important**: Run the training command from the `Train/` directory:

```bash
# Navigate to Train directory
cd Train

# Activate your virtual environment if not already active
source ../venv/bin/activate  # Adjust path if venv is elsewhere

# Run training
axolotl train finetune_gpt_oss_20b.yaml
```

**For debugging** (if you encounter errors):

```bash
cd Train
chmod +x debug_train.sh
./debug_train.sh
```

This will show more detailed error messages and verify dataset paths.

### Monitor Training

Training progress will be displayed in the terminal. You can also monitor:
- Loss values (training and validation)
- GPU utilization (`nvidia-smi`)
- Checkpoints saved in `./outputs/gpt-oss-20b-creative-writing/`

### Resume from Checkpoint

If training is interrupted, you can resume:

```bash
axolotl train finetune_gpt_oss_20b.yaml --resume_from_checkpoint ./outputs/gpt-oss-20b-creative-writing/checkpoint-XXXX
```

## Post-Training

### Test the Fine-Tuned Model

After training completes, LoRA adapters will be saved in `./lora_outputs/gpt-oss-20b-creative-writing/`. You can test the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./lora_outputs/gpt-oss-20b-creative-writing/")

# Test generation
prompt = "The old lighthouse stood"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Merge LoRA Adapters (Optional)

To create a standalone model with LoRA weights merged:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned",
    torch_dtype=torch.bfloat16
)

lora_model = PeftModel.from_pretrained(base_model, "./lora_outputs/gpt-oss-20b-creative-writing/")

# Merge and save
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./merged_gpt-oss-20b-creative-writing/")
```

## Troubleshooting

### Training Errors (ChildFailedError / Rank Failures)

If you get errors like "ChildFailedError" or rank failures:

1. **Check dataset paths**: Ensure paths in the YAML are correct relative to where you run the command
   - If running from `Train/`, paths should be `../Datasets/...`
   - If running from root, paths should be `Datasets/...`

2. **Use the debug script**:
   ```bash
   cd Train
   chmod +x debug_train.sh
   ./debug_train.sh
   ```

3. **Try the simplified config**: If the dual-dataset config fails, try:
   ```bash
   axolotl train finetune_gpt_oss_20b_simple.yaml
   ```
   This uses a single dataset with automatic validation split.

4. **Update Transformers**: If you get "model type `gpt_oss` not recognized" error:
   ```bash
   # Update transformers to latest version (supports GPT-OSS)
   pip install --upgrade transformers
   
   # Or install from source for absolute latest:
   pip install git+https://github.com/huggingface/transformers.git
   ```
   Newer versions of transformers fully support GPT-OSS models.

5. **Check model path**: Verify the abliterated model exists at:
   `/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned`

5. **Enable verbose logging**: Check the logs in `training.log` after running `debug_train.sh`

6. **Check GPU availability**: 
   ```bash
   nvidia-smi
   python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
   ```

7. **Check Axolotl config format**: Ensure YAML syntax is correct (no tabs, proper indentation)

### Out of Memory (OOM) Errors

1. **Reduce batch size**: Decrease `per_device_train_batch_size` to 1
2. **Increase gradient accumulation**: Increase `gradient_accumulation_steps` to maintain effective batch size
3. **Reduce sequence length**: Decrease `max_seq_length` to 1024 or 512
4. **Use DeepSpeed**: Switch to DeepSpeed ZeRO Stage 2 for RTX 3090s
5. **Enable gradient checkpointing**: Already enabled by default (`gradient_checkpointing: true`)

### Slow Training

1. **Use multiple GPUs**: Training automatically uses all available GPUs
2. **Increase batch size**: If you have VRAM headroom
3. **Reduce sequence length**: Shorter sequences train faster
4. **Disable evaluation during training**: Set `eval_strategy: no` temporarily

### Model Not Improving

1. **Increase epochs**: Try 4-5 epochs instead of 3
2. **Adjust learning rate**: Try different learning rates (1e-4 to 5e-4)
3. **Increase LoRA rank**: Try `lora_r: 16` or `lora_r: 32`
4. **Check dataset quality**: Ensure your creative writing dataset is diverse and high-quality

## Best Practices

1. **Start Small**: Begin with default settings, then adjust based on results
2. **Monitor Validation Loss**: Stop training if validation loss stops decreasing
3. **Save Checkpoints**: Regularly save checkpoints to avoid losing progress
4. **Test Frequently**: Test the model during training to see qualitative improvements
5. **Preserve Abliteration**: LoRA preserves your abliteration work - full fine-tuning would risk losing it

## Why LoRA Instead of Full Fine-Tuning?

1. **Preserves Abliteration**: Full fine-tuning risks overwriting the abliteration modifications
2. **Memory Efficient**: LoRA only trains ~1-5% of parameters
3. **Faster Training**: Significantly faster than full fine-tuning
4. **Flexible**: Can easily swap/combine different LoRA adapters
5. **Best Practice**: Industry standard for fine-tuning large models

## Additional Resources

- [Axolotl Documentation](https://github.com/axolotl-ai/axolotl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## File Structure

```
.
├── finetune_gpt_oss_20b.yaml      # Main Axolotl configuration
├── deepspeed_config.json          # DeepSpeed config (for RTX 3090s)
├── split_dataset.py               # Dataset splitting script
├── README_FINETUNING.md          # This file
├── Datasets/
│   ├── dataset_20251128_154142.jsonl
│   ├── dataset_20251128_154142_train.jsonl
│   └── dataset_20251128_154142_val.jsonl
├── outputs/                       # Training outputs and checkpoints
│   └── gpt-oss-20b-creative-writing/
└── lora_outputs/                  # LoRA adapter weights
    └── gpt-oss-20b-creative-writing/
```

## Notes

- The abliterated model must be in HuggingFace format (which it is, from `sharded_ablate.py`)
- Original model weights remain frozen during LoRA training
- LoRA adapters can be merged back into the model or kept separate for flexibility
- The `{"text":"STORY"}` format is perfect for CLM - no restructuring needed

