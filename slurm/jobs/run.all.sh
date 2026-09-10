DATASETS=("moons" "skin-cancer")
MODELS=("dann" "cida" "pcida")

# Run baseline
for dataset in "${DATASETS[@]}"; do
    sbatch --job-name=${dataset}_baseline baseline.job $dataset
done

MODES=("fixed" "progressive")

# Run DANN, CIDA, and PCIDA
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            sbatch --job-name=${dataset}_${model}_${mode} da.job $dataset $model $mode
        done
    done
done