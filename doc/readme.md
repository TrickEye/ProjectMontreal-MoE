## simple_rseft/ — 6 files, ~230 lines avg each

| File | Purpose | Key Content |
| --- | --- | --- |
| model_utils.py | Model loading | 4-bit QLoRA, chat template, seed setting |
| data_utils.py | Data pipeline | GSM8K loading from HF, prompt formatting, tokenization with SFT loss masking |
| hook_rfar.py | Core RFAR algorithm | entropy → forking tokens → expert activation → RFAR scores → expert selection |
| train.py | Training | Stage 1 (router-only LoRA) + Stage 3 (expert-selective LoRA) via PEFT |
| evaluate.py | Evaluation | GSM8K answer extraction (regex priority chain), accuracy scoring |
| main.py | Orchestrator | Full pipeline with CLI flags to skip individual stages |

## Pipeline flow

```
main.py
  ├─ load GSM8K (openai/gsm8k)
  ├─ Stage 1: train_stage1_router()  → LoRA on mlp.gate → outputs/stage1_router/
  ├─ Stage 2: run_rfar_analysis()     → RFAR[16×64] → outputs/rfar_results.json
  ├─ Stage 3: train_stage3_experts()  → LoRA on selected experts → outputs/stage3_experts/
  └─ Evaluate: base vs router-tuned vs expert-tuned accuracy
```

## How to run
```
pip install -f requirements.txt
cd simple_rseft

# Quick smoke test (200 samples)
python main.py --model_path deepseek-ai/deepseek-moe-16b-chat --max_samples 200

## Full run
python main.py --model_path deepseek-ai/deepseek-moe-16b-chat

# Skip stages you already ran
python main.py --model_path <path> --skip_stage1 --skip_stage2
```
