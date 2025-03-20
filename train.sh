#!/usr/bin/env bash
# Initialize variables as empty
GPU=""
MODE=""
CORR="0"  # Default value
GCRS_ENT="20.0"  # Default value
GCLST="0.1"  # Default value
GSEP="-0.001"  # Default value
GL1="0.0"  # Default value
GORTHO="0.0"  # Default value
ICRS_ENT="20.0"  # Default value
ICLST="0.1"  # Default value
ISEP="-0.001"  # Default value
IL1="0.0"  # Default value
IORTHO="0.0"  # Default value
RUN_NAME=""  # Default is empty to use the default in main.py

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
        --corr)
            CORR="$2"
            shift 2
            ;;
        --gortho)
            GORTHO="$2"
            shift 2
            ;;
        --iortho)
            IORTHO="$2"
            shift 2
            ;;
        --gcrs_ent)
            GCRS_ENT="$2"
            shift 2
            ;;
        --gclst)
            GCLST="$2"
            shift 2
            ;;
        --gsep)
            GSEP="$2"
            shift 2
            ;;
        --gl1)
            GL1="$2"
            shift 2
            ;;
        --icrs_ent)
            ICRS_ENT="$2"
            shift 2
            ;;
        --iclst)
            ICLST="$2"
            shift 2
            ;;
        --isep)
            ISEP="$2"
            shift 2
            ;;
        --il1)
            IL1="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
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
    echo "Usage: sbatch train.sh --gpu [a5000|a6000] --mode [image|genetics|multi] [--corr VALUE] [--gortho VALUE] [--iortho VALUE] [--gcrs_ent VALUE] [--gclst VALUE] [--gsep VALUE] [--gl1 VALUE] [--icrs_ent VALUE] [--iclst VALUE] [--isep VALUE] [--il1 VALUE] [--run-name NAME]"
    exit 1
fi

# Create a temporary script with the correct SBATCH arguments
TEMP_SCRIPT=$(mktemp)
cat > $TEMP_SCRIPT << EOT
#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=3:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:${GPU}:1
#SBATCH --output=logs/${MODE}/%j.out

eval "\$(conda shell.bash hook)"
conda activate intnn

python main.py --configs configs/${MODE}.yaml \
    --corr ${CORR} \
    --gortho ${GORTHO} \
    --iortho ${IORTHO} \
    --gcrs_ent ${GCRS_ENT} \
    --gclst ${GCLST} \
    --gsep ${GSEP} \
    --gl1 ${GL1} \
    --icrs_ent ${ICRS_ENT} \
    --iclst ${ICLST} \
    --isep ${ISEP} \
    --il1 ${IL1} \
    ${RUN_NAME_ARG}
EOT

# Submit the temporary script and then remove it
sbatch $TEMP_SCRIPT
rm $TEMP_SCRIPT
