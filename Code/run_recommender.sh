#!/usr/bin/env bash

# Script de ayuda
function usage() {
  echo "Uso: $0 [-h] -m método -n clusters [-d dataset]" >&2
  echo
  echo "Opciones:" >&2
  echo "  -h            Muestra esta ayuda." >&2
  echo "  -m método     Tipo de clustering: 'hard'/'fuzzy'/'hac'." >&2
  echo "  -n clusters   Número de clusters." >&2
  echo "  -d dataset    Nombre del CSV en ../Datasets (sin .csv). Default: ml-small" >&2
  exit 1
}

# Variables (sin valores por defecto para método ni clusters)
method=""
n_clusters=""
dataset="ml-small"

# Parseo de opciones
while getopts ":hm:n:d:" opt; do
  case ${opt} in
    h ) usage ;;
    m ) method=$OPTARG ;;
    n ) n_clusters=$OPTARG ;;
    d ) dataset=$OPTARG ;;
    \? ) echo "Opción inválida: -$OPTARG" >&2; usage ;;
    : ) echo "La opción -$OPTARG requiere un argumento." >&2; usage ;;
  esac
done

# Validaciones obligatorias
if [[ -z "$method" || -z "$n_clusters" ]]; then
  echo "Error: Las opciones -m y -n son obligatorias." >&2
  usage
fi

# Ejecución del script Python
python3 recommender_clustering.py -m "$method" -n "$n_clusters" -d "$dataset"
