#!/usr/bin/env bash

# Initialize variables as empty
GPU=""
MODE=""

# Parse command line arguments first
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$GPU" ] || [ -z "$MODE" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch train_blackbox.sh --gpu [a5000|a6000] --mode [image|genetics]"
    exit 1
fi

# Create a temporary script with the correct SBATCH arguments
TEMP_SCRIPT=$(mktemp)
cat > $TEMP_SCRIPT << EOT
#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:${GPU}:1
#SBATCH --output=logs/${MODE}_backbone/%j.out

eval "\$(conda shell.bash hook)"
conda activate intnn
python3 train_blackbox.py --configs configs/${MODE}_backbone.yaml
EOT

# Submit the temporary script and then remove it
sbatch $TEMP_SCRIPT
rm $TEMP_SCRIPT
