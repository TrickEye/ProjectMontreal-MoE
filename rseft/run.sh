#!/bin/bash
# =============================================================================
# R-SEFT: Reasoning-Specialized Expert Fine-Tuning
# Launch script for OLMoE-1B-7B on GSM8K
# =============================================================================

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./rseft_output}"
TOP_K="${TOP_K:-64}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-2}"
PHASE3_EPOCHS="${PHASE3_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ENTROPY_PERCENTILE="${ENTROPY_PERCENTILE:-0.8}"

echo "=============================================="
echo "  R-SEFT: OLMoE-1B-7B → GSM8K"
echo "=============================================="
echo "  Output dir:        $OUTPUT_DIR"
echo "  Top-K experts:     $TOP_K"
echo "  Phase 1 epochs:    $PHASE1_EPOCHS"
echo "  Phase 3 epochs:    $PHASE3_EPOCHS"
echo "  Batch size:        $BATCH_SIZE"
echo "  Entropy %ile:      $ENTROPY_PERCENTILE"
echo "=============================================="

# Install dependencies (uncomment if needed)
# pip install -r requirements.txt

# Run the full R-SEFT pipeline
python main.py \
    --output_dir "$OUTPUT_DIR" \
    --top_k_experts "$TOP_K" \
    --phase1_epochs "$PHASE1_EPOCHS" \
    --phase3_epochs "$PHASE3_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --entropy_percentile "$ENTROPY_PERCENTILE" \
    "$@"

echo ""
echo "Done! Results saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate only:"
echo "  python main.py --eval_only $OUTPUT_DIR/phase3_final"
echo ""
echo "To resume from Phase 1 checkpoint:"
echo "  python main.py --skip_phase1 --output_dir $OUTPUT_DIR"
echo ""
echo "To resume with existing expert selection:"
echo "  python main.py --skip_phase1 --skip_phase2 --output_dir $OUTPUT_DIR"
