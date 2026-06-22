#!/bin/bash
#SBATCH --job-name=fastchess-resume
#SBATCH --account=kwf@v100              # Allocation V100 de votre projet [cite: 106]
#SBATCH --time=20:00:00                 # [cite: 106]
#SBATCH --nodes=1                       # [cite: 106]
#SBATCH --ntasks-per-node=1             # [cite: 106]
#SBATCH --cpus-per-task=10              # [cite: 106]
#SBATCH --gres=gpu:1                    # [cite: 106]
#SBATCH -C v100-32g                     # [cite: 106]
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

# --- SÉCURITÉ : Vérification de l'argument unique ---
if [ "$#" -ne 1 ]; then
    echo "Erreur : Nombre d'arguments incorrect."
    echo "Utilisation : sbatch resume_tournament.slurm <Nom_Tournoi>"
    exit 1
fi

TOURNAMENT_NAME=$1

# --- CONFIGURATION DES CHEMINS JEAN-ZAY ---
BASE_DIR="/lustre/fsn1/projects/rech/kwf/uzr96yg/leela" # [cite: 109]
CONFIGS_DIR="${BASE_DIR}/configs"                       # [cite: 109]
FASTCHESS_EXEC="${BASE_DIR}/fastchess/fastchess"       # [cite: 110]
CONFIG_FILE="${CONFIGS_DIR}/${TOURNAMENT_NAME}.json"

# Vérification de sécurité pour s'assurer que le fichier JSON du tournoi existe
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Erreur : Le fichier de configuration pour reprendre ce tournoi est introuvable :"
    echo " -> $CONFIG_FILE"
    exit 1
fi

echo "=== Reprise du tournoi : ${TOURNAMENT_NAME} ==="
echo "Fichier de configuration trouvé : $(basename ${CONFIG_FILE})"

# --- CHARGEMENT DES MODULES ---
module purge
module load cuda/12.8.0                 # [cite: 109]
module load gcc/13.3.0                  # [cite: 109]

echo "--------------------------------------------------"
echo "Relancement de fastchess (Logs appended to logs/${TOURNAMENT_NAME}.out/.err)..."
echo "--------------------------------------------------"

# EXÉCUTION : Reprise à l'aide de l'opérateur '>>' pour concaténer les logs
"$FASTCHESS_EXEC" -config file="$CONFIG_FILE" >> "logs/${TOURNAMENT_NAME}.out" 2>> "logs/${TOURNAMENT_NAME}.err"

echo "Session de reprise terminée."