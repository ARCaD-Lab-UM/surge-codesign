#!/bin/bash
# Batch runner for codesign experiments with seed sweep
# Usage: ./run_batch.sh [script_name] [num_runs] [start_seed]
# Examples:
#   ./run_batch.sh run_cma_codesign.py 5          # seeds 1,2,3,4,5
#   ./run_batch.sh run_injected_es.py 10 100      # seeds 100,101,...,109

SCRIPT_NAME=${1}  # Required: script name (e.g., run_cma_codesign.py)
NUM_RUNS=${2:-5}  # Default to 5 runs if not specified
START_SEED=${3:-1} # Default start seed = 1 (avoid 0, pycma treats it as "no seed")

# Extract script base name (without .py extension)
SCRIPT_BASE=$(basename "$SCRIPT_NAME" .py)
SCRIPT_PATH="scripts/${SCRIPT_NAME}"

# Generate batch ID and directory
BATCH_ID=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="logs/experiments/batch_${SCRIPT_BASE}_${BATCH_ID}"
BATCH_LOG="${BATCH_DIR}/batch_info.txt"

echo "=============================================="
echo "Codesign Batch Runner (Seed Sweep)"
echo "=============================================="
echo "Script: ${SCRIPT_PATH}"
echo "Number of runs: ${NUM_RUNS}"
echo "Seeds: ${START_SEED} .. $((START_SEED + NUM_RUNS - 1))"
echo "Batch ID: ${BATCH_ID}"
echo "Batch directory: ${BATCH_DIR}"
echo "=============================================="

# Create batch directory
mkdir -p "${BATCH_DIR}"

# Create batch log file
echo "Batch ID: ${BATCH_ID}" > "$BATCH_LOG"
echo "Script: ${SCRIPT_NAME}" >> "$BATCH_LOG"
echo "Number of runs: ${NUM_RUNS}" >> "$BATCH_LOG"
echo "Seeds: ${START_SEED} .. $((START_SEED + NUM_RUNS - 1))" >> "$BATCH_LOG"
echo "Started at: $(date)" >> "$BATCH_LOG"
echo "" >> "$BATCH_LOG"
echo "Run directories:" >> "$BATCH_LOG"

# Run experiments
for i in $(seq 1 $NUM_RUNS); do
    SEED=$((START_SEED + i - 1))
    echo ""
    echo "=============================================="
    echo "Starting run $i / $NUM_RUNS  (seed=${SEED})"
    echo "=============================================="

    # Run the experiment with this seed
    python "$SCRIPT_PATH" --seed $SEED
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Run $i failed with exit code $EXIT_CODE"
        echo "Run $i: FAILED (exit code $EXIT_CODE)" >> "$BATCH_LOG"
    else
        # Find the most recent run directory matching the script type
        # Map script name -> DataLogger run_name pattern
        if [[ "$SCRIPT_BASE" == "run_cma_codesign" ]]; then
            PATTERN="hopper_codesign_cmaes_*"
        elif [[ "$SCRIPT_BASE" == "run_meanshift_es" ]]; then
            PATTERN="hopper_codesign_meanshift_*"
        elif [[ "$SCRIPT_BASE" == "run_injected_es" ]]; then
            PATTERN="hopper_codesign_cma_inject_*"
        elif [[ "$SCRIPT_BASE" == "run_codesign" ]]; then
            # Must not match the more specific patterns above
            PATTERN="hopper_codesign_[0-9]*"
        else
            PATTERN="hopper_codesign_*"
        fi

        LATEST_RUN=$(ls -td logs/optim/${PATTERN} 2>/dev/null | head -1)
        
        if [ -n "$LATEST_RUN" ]; then
            # Move the run to batch directory
            RUN_NAME=$(basename "$LATEST_RUN")
            mv "$LATEST_RUN" "${BATCH_DIR}/"
            echo "Run $i completed: ${BATCH_DIR}/${RUN_NAME}"
            echo "Run $i (seed=${SEED}): ${RUN_NAME}" >> "$BATCH_LOG"
        else
            echo "Run $i: Could not find output directory" >> "$BATCH_LOG"
        fi
    fi
done

echo "" >> "$BATCH_LOG"
echo "Finished at: $(date)" >> "$BATCH_LOG"

echo ""
echo "=============================================="
echo "Batch completed!"
echo "Results saved to: ${BATCH_DIR}"
echo "=============================================="
