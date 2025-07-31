#!/bin/bash

# Bootstrap Heatmap Generation Pipeline
# Generate heatmap plots for all patients, slices, and modes

echo "🎨 Starting Bootstrap Heatmap Generation Pipeline"
echo "================================================="

# Configuration
RESULTS_DIR="results_dmg_fit_goodinit"
OUTPUT_DIR="results_dmg_fit_goodinit_trajectories"
DATA_PATH="data"
PYTHON_SCRIPT="plot_pretty_gc.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found"
    echo "Please make sure the plotting script is in the current directory"
    exit 1
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory '$RESULTS_DIR' not found"
    echo "Please make sure you have run the bootstrap analysis first"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_PATH" ]; then
    echo "⚠️  Warning: Data directory '$DATA_PATH' not found"
    echo "Original data overlay will not be available"
fi

# Analysis modes
MODES=("all" "train")

# Counters
echo "Configuration:"
echo "  Results directory: $RESULTS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data path: $DATA_PATH"
echo "  Python script: $PYTHON_SCRIPT"
echo "  Modes: ${MODES[@]}"
echo ""

# Main processing loop
echo "🔍 Scanning for bootstrap prediction files..."
echo ""

# Debug: Show the actual directory structure first
echo "🔍 Debugging directory structure..."
echo "Looking in: $RESULTS_DIR"
if [ -d "$RESULTS_DIR" ]; then
    echo "📂 Found result directories:"
    for patient_dir in "$RESULTS_DIR"/*/; do
        if [ -d "$patient_dir" ]; then
            patient_id=$(basename "$patient_dir")
            echo "  👤 $patient_id"
            
            unified_dir="$patient_dir/unified"
            if [ -d "$unified_dir" ]; then
                echo "    📁 unified/"
                for slice_dir in "$unified_dir"/*/; do
                    if [ -d "$slice_dir" ]; then
                        slice_type=$(basename "$slice_dir")
                        echo "      📁 $slice_type/"
                        
                        # List CSV files in this directory
                        csv_count=$(find "$slice_dir" -name "*.csv" -type f | wc -l)
                        if [ $csv_count -gt 0 ]; then
                            echo "        📄 CSV files found:"
                            find "$slice_dir" -name "*.csv" -type f -exec basename {} \; | head -3
                        else
                            echo "        ❌ No CSV files found"
                        fi
                    fi
                done
            else
                echo "    ❌ No unified directory"
            fi
        fi
    done
else
    echo "❌ Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo ""
echo "🚀 Starting file processing..."

# Initialize counters outside the subshell
total_found=0
total_processed=0

# Create a temporary file to store results
temp_results=$(mktemp)

# Find all unified CSV files and process them
for patient_dir in "$RESULTS_DIR"/*/; do
    if [ ! -d "$patient_dir" ]; then
        continue
    fi
    
    patient_id=$(basename "$patient_dir")
    unified_dir="$patient_dir/unified"
    
    if [ ! -d "$unified_dir" ]; then
        echo "⚠️  No unified directory found for $patient_id"
        continue
    fi
    
    echo ""
    echo "👤 Processing patient: $patient_id"
    
    # Process each slice type directory
    for slice_dir in "$unified_dir"/*/; do
        if [ ! -d "$slice_dir" ]; then
            continue
        fi
        
        slice_type=$(basename "$slice_dir")
        
        # The correct filename format: patient-id_slice-id_bootstrap_predictions_unified.csv
        # Examples: PIDz035_volume_bootstrap_predictions_unified.csv
        #          PIDz035_largest_0_bootstrap_predictions_unified.csv
        csv_file="$slice_dir/${patient_id}_${slice_type}_bootstrap_predictions_unified.csv"
        
        echo "  🔍 Looking for: $(basename "$csv_file")"
        
        # If the expected file doesn't exist, try to find any bootstrap predictions file
        if [ ! -f "$csv_file" ]; then
            # Look for any bootstrap predictions file in this directory
            alternative_file=$(find "$slice_dir" -name "*bootstrap_predictions_unified.csv" -type f | head -1)
            if [ -n "$alternative_file" ]; then
                echo "    ℹ️  Expected file not found, but found: $(basename "$alternative_file")"
                csv_file="$alternative_file"
            else
                echo "    ⚠️  File not found: $csv_file"
                # Debug: List what files actually exist in this directory
                echo "    📂 Files in directory:"
                if [ -d "$slice_dir" ]; then
                    ls -la "$slice_dir" | grep -E "\.(csv|CSV)$" || echo "    No CSV files found"
                fi
                # Try alternative file patterns
                echo "    🔍 Searching for alternative patterns..."
                find "$slice_dir" -name "*bootstrap_predictions*" -type f 2>/dev/null || echo "    No bootstrap prediction files found"
                continue
            fi
        fi
        
        if [ ! -r "$csv_file" ]; then
            echo "    ❌ Cannot read file: $csv_file"
            continue
        fi
        
        echo "    📊 Found CSV: $(basename "$csv_file")"
        echo "    Patient: $patient_id, Slice: $slice_type"
        
        ((total_found++))
        
        # Process both modes for this CSV
        for mode in "${MODES[@]}"; do
            echo "    🎨 Generating $mode mode plot..."
            
            # Create patient-specific output directory
            patient_output_dir="$OUTPUT_DIR/$patient_id"
            mkdir -p "$patient_output_dir"
            
            # Run Python plotting script
            if python "$PYTHON_SCRIPT" \
                --csv_path "$csv_file" \
                --output_path "$patient_output_dir" \
                --mode "$mode" \
                --data_path "$DATA_PATH" 2>&1; then
                
                echo "      ✅ Success: $patient_id - $slice_type - $mode"
                echo "SUCCESS" >> "$temp_results"
            else
                echo "      ❌ Failed: $patient_id - $slice_type - $mode"
                echo "FAILED" >> "$temp_results"
            fi
            
            ((total_processed++))
        done
    done
done

# Count results
successful_plots=$(grep -c "SUCCESS" "$temp_results" 2>/dev/null || echo "0")
failed_plots=$(grep -c "FAILED" "$temp_results" 2>/dev/null || echo "0")
total_plots=$((successful_plots + failed_plots))

# Clean up
rm -f "$temp_results"

# Wait for all background processes to complete
wait

echo ""
echo "🎉 Heatmap generation pipeline completed!"
echo "========================================"
echo ""
echo "📊 Generation Summary:"
echo "  Total plots attempted: $total_plots"
echo "  Successful plots: $successful_plots"
echo "  Failed plots: $failed_plots"
if [ $total_plots -gt 0 ]; then
    success_rate=$(( (successful_plots * 100) / total_plots ))
    echo "  Success rate: ${success_rate}%"
fi
echo ""
echo "📁 Output structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── [patient_id]/"
echo "  │   ├── [patient]_[slice]_all_heatmap.png"
echo "  │   ├── [patient]_[slice]_all_heatmap.pdf"
echo "  │   ├── [patient]_[slice]_train_heatmap.png"
echo "  │   └── [patient]_[slice]_train_heatmap.pdf"
echo "  └── ..."
echo ""

# Generate summary of what was processed
echo "📋 Processing Summary:"
find "$OUTPUT_DIR" -name "*.png" | wc -l | xargs echo "  PNG files created:"
find "$OUTPUT_DIR" -name "*.pdf" | wc -l | xargs echo "  PDF files created:"

# List patients processed
echo ""
echo "👥 Patients processed:"
find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read patient_dir; do
    patient=$(basename "$patient_dir")
    plot_count=$(find "$patient_dir" -name "*.png" | wc -l)
    echo "  - $patient ($plot_count plots)"
done

echo ""
echo "✨ All heatmap plots have been generated!"
echo "Check the '$OUTPUT_DIR' directory for your results."