#!/bin/bash

# Load anaconda module
module load anaconda3/2024.06

# Deactivate any existing environments
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

# Activate the project environment with ABSOLUTE PATH
conda activate /scratch/kulkarni.vedan/drug_discovery_ml/env

echo "✅ Environment activated!"
echo "Python: $(which python)"
echo "Environment: /scratch/kulkarni.vedan/drug_discovery_ml/env"
