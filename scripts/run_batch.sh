#!/bin/bash
# Batch runner for codesign experiments
# Usage: ./run_batch.sh [script_name] [num_runs]
# Examples:
#   ./run_batch.sh run_cma_codesign.py 5     # Run cma 5 times
#   ./run_batch.sh run_guided_es_codesign.py 20

SCRIPT_NAME=${1}  # Required: script name (e.g., run_guided_es_codesign.py)
NUM_RUNS=${2:-5}  # Default to 5 runs if not specified

# Extract script base name (without .py extension)
SCRIPT_BASE=$(basename "$SCRIPT_NAME" .py)
SCRIPT_PATH="scripts/${SCRIPT_NAME}"

# Generate batch ID and directory
BATCH_ID=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="logs/experiments/batch_${SCRIPT_BASE}_${BATCH_ID}"
BATCH_LOG="${BATCH_DIR}/batch_info.txt"

echo "=============================================="
echo "Codesign Batch Runner"
echo "=============================================="
echo "Script: ${SCRIPT_PATH}"
echo "Number of runs: ${NUM_RUNS}"
echo "Batch ID: ${BATCH_ID}"
echo "Batch directory: ${BATCH_DIR}"
echo "=============================================="

# Create batch directory
mkdir -p "${BATCH_DIR}"

# Create batch log file
echo "Batch ID: ${BATCH_ID}" > "$BATCH_LOG"
echo "Script: ${SCRIPT_NAME}" >> "$BATCH_LOG"
echo "Number of runs: ${NUM_RUNS}" >> "$BATCH_LOG"
echo "Started at: $(date)" >> "$BATCH_LOG"
echo "" >> "$BATCH_LOG"
echo "Run directories:" >> "$BATCH_LOG"

# Run experiments
for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "=============================================="
    echo "Starting run $i / $NUM_RUNS"
    echo "=============================================="
    
    # Run the experiment
    python "$SCRIPT_PATH"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Run $i failed with exit code $EXIT_CODE"
        echo "Run $i: FAILED (exit code $EXIT_CODE)" >> "$BATCH_LOG"
    else
        # Find the most recent run directory matching the script type
        if [[ "$SCRIPT_BASE" == *"cma"* ]]; then
            PATTERN="hopper_codesign_cmaes_*"
        elif [[ "$SCRIPT_BASE" == *"guided"* ]]; then
            PATTERN="hopper_codesign_guided_es_*"
        else
            PATTERN="hopper_codesign_*"
        fi
        
        LATEST_RUN=$(ls -td logs/optim/${PATTERN} 2>/dev/null | head -1)
        
        if [ -n "$LATEST_RUN" ]; then
            # Move the run to batch directory
            RUN_NAME=$(basename "$LATEST_RUN")
            mv "$LATEST_RUN" "${BATCH_DIR}/"
            echo "Run $i completed: ${BATCH_DIR}/${RUN_NAME}"
            echo "Run $i: ${RUN_NAME}" >> "$BATCH_LOG"
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
