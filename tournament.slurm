#!/bin/bash
#SBATCH --job-name=fastchess-tournament
#SBATCH --account=kwf@v100              # Allocation V100 de votre projet 
#SBATCH --time=20:00:00                 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10              
#SBATCH --gres=gpu:1                    
#SBATCH -C v100-32g                     # 
#SBATCH --output=logs/tournament_%j.out
#SBATCH --error=logs/tournament_%j.err

# --- SÉCURITÉ : Vérification des arguments minimaux ---
if [ "$#" -lt 3 ]; then
    echo "Erreur : Nombre d'arguments insuffisant."
    echo "Utilisation : sbatch tournament.slurm <Nom_Tournoi> <Modèle_1> <Modèle_2> ... <Modèle_N>"
    exit 1
fi

# Extraction du premier argument pour le nom du tournoi
TOURNAMENT_NAME=$1
shift # Décale les arguments pour isoler la liste des modèles

# --- CONFIGURATION DES CHEMINS JEAN-ZAY ---
BASE_DIR="/lustre/fsn1/projects/rech/kwf/uzr96yg/leela" # 
WORK_DIR="/lustre/fswork/projects/rech/kwf/uzr96yg/lc0"
LC0_EXEC="${WORK_DIR}/build/release/lc0"
BOOKS_DIR="${BASE_DIR}/books"
PGNS_DIR="${BASE_DIR}/pgns"
CONFIGS_DIR="${BASE_DIR}/configs"

# Création des dossiers de sortie s'ils n'existent pas 
mkdir -p logs "${PGNS_DIR}" "${CONFIGS_DIR}"

echo "=== Préparation du tournoi : ${TOURNAMENT_NAME} ==="

# --- CHARGEMENT DES MODULES ---
module purge
module load cuda/12.8.0                 # 
module load gcc/13.3.0                  # 

# Initialisation du tableau de commande fastchess
# 🌟 MODIFICATION 2 : Pointage vers le nouveau dossier de fastchess
FASTCHESS_CMD=(
    "${BASE_DIR}/fastchess/fastchess"
    -event "${TOURNAMENT_NAME}"
)

# --- BOUCLE DYNAMIQUE : Traitement de chaque modèle ---
for MODEL_NAME in "$@"; do
    NETWORKS_DIR="${BASE_DIR}/networks/${MODEL_NAME}" # 
    
    # Extraction de la version de poids la plus récente 
    SELECTED_WEIGHT=$(python3 -c "
import os, re, sys

folder = '$NETWORKS_DIR'
if not os.path.exists(folder):
    sys.exit(1)

files = [f for f in os.listdir(folder) if f.endswith('.pb.gz')]
if not files:
    sys.exit(2)

numeric_files = []
for f in files:
    match = re.search(r'(\d+)\.pb\.gz$', f)
    if match:
        numeric_files.append((int(match.group(1)), f))

if numeric_files:
    numeric_files.sort(key=lambda x: x[0], reverse=True)
    print(os.path.join(folder, numeric_files[0][1]))
else:
    files.sort()
    print(os.path.join(folder, files[0]))
" 2>/dev/null) # 

    # Gestion des erreurs de l'extraction 
    exit_status=$?
    if [ $exit_status -eq 1 ]; then
        echo "Erreur : Le dossier du modèle '${NETWORKS_DIR}' n'existe pas."
        exit 1
    elif [ $exit_status -eq 2 ] || [ -z "${SELECTED_WEIGHT}" ]; then
        echo "Erreur : Aucun fichier de poids '.pb.gz' trouvé dans ${NETWORKS_DIR}."
        exit 1
    fi

    echo " -> Modèle enregistré : ${MODEL_NAME} | Poids : $(basename ${SELECTED_WEIGHT})"
    
    # Injection dynamique des arguments du moteur
    FASTCHESS_CMD+=( -engine name="${MODEL_NAME}" args="-w ${SELECTED_WEIGHT}" )
done

FASTCHESS_CMD+=(
    -openings file="${BOOKS_DIR}/book-6-ply-unbalanced.pgn" format=pgn order=sequential
    -each cmd="${LC0_EXEC}" tc=60+0.1
    -rounds 400
    -concurrency 1
    -pgnout file="${PGNS_DIR}/${TOURNAMENT_NAME}.pgn"
    -config outname="${CONFIGS_DIR}/${TOURNAMENT_NAME}.json"
)

echo "--------------------------------------------------"
echo "Lancement de fastchess (Logs redirected to logs/${TOURNAMENT_NAME}.out/.err)..."
echo "--------------------------------------------------"

"${FASTCHESS_CMD[@]}" > "logs/${TOURNAMENT_NAME}.out" 2> "logs/${TOURNAMENT_NAME}.err"

echo "Tournoi terminé avec succès !"