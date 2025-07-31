#!/bin/bash
echo "🚀 Starting DMG Bootstrap Analysis Pipeline"
echo "============================================"

# Create output directory
OUTPUT_FOLDER="debug"
mkdir -p $OUTPUT_FOLDER

# Configuration
BOOTSTRAP_SAMPLES=30
N_JOBS=10

# Patient list (valid patients only)
##################################
# no-volume model patients
# PATIENTS=(
#     "PIDz057" 
#     "PIDz140"
#     # "PIDz160"
#     "PIDz161"
#     "PIDz224"
#     # "PIDz254"
# )

###############################
# full model patients
PATIENTS=(
    "PIDz035"
    "PIDz069"
    # "PIDz074"
    # "PIDz077"
    # "PIDz279"
)
# PATIENTS=(
#     "PIDz035"
# )

# Noise levels to test
NOISE_LEVELS=(0.1)

# Data types to analyze
DATA_TYPES=("volume")

# Add slice data types
for i in {0..5}; do
    DATA_TYPES+=("largest_$i")
    DATA_TYPES+=("initial_$i")
done

echo "Configuration:"
echo "  Bootstrap samples: $BOOTSTRAP_SAMPLES"
echo "  Noise levels: ${NOISE_LEVELS[@]}"
echo "  Parallel jobs: $N_JOBS"
echo "  Patients: ${#PATIENTS[@]}"
echo "  Data types: ${#DATA_TYPES[@]} (${DATA_TYPES[@]})"
echo ""

# Analysis counter
total_analyses=$((${#PATIENTS[@]} * ${#DATA_TYPES[@]} * ${#NOISE_LEVELS[@]}))
current_analysis=0

# Run bootstrap analysis for each combination
for patient in "${PATIENTS[@]}"; do
    mkdir -p "${OUTPUT_FOLDER}/${patient}"
    for data_type in "${DATA_TYPES[@]}"; do
        for noise_level in "${NOISE_LEVELS[@]}"; do
            current_analysis=$((current_analysis + 1))
            
            echo "[$current_analysis/$total_analyses] Analyzing $patient - $data_type - noise_$noise_level"
            
            # Run bootstrap analysis
            python run_dmg_full.py \
                --patient "$patient" \
                --data_type "$data_type" \
                --n_bootstrap $BOOTSTRAP_SAMPLES \
                --noise_level $noise_level \
                --n_jobs $N_JOBS \
                --output_dir "${OUTPUT_FOLDER}" \
                --bootstrap_path "results_dmg_fit" \
                --normalization
                # --no_volume
            
            if [ $? -eq 0 ]; then
                echo "  ✅ Success: $patient - $data_type - noise_$noise_level"
            else
                echo "  ❌ Failed: $patient - $data_type - noise_$noise_level"
            fi
            echo ""
        done
    done
done