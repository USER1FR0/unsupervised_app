"""CLI para generar datasets sinteticos.

Uso:
    python scripts/generate_synthetic.py --dataset synthetic --n 500 --seed 42
    python scripts/generate_synthetic.py --dataset demo --n 100 --seed 99
"""
import argparse
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_synthetic_dataset, print_validation_stats


DATA_DIR = ROOT / "data"
ALLOWED_DATASETS = {"synthetic", "demo"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador de dataset sintetico")
    parser.add_argument("--dataset", choices=sorted(ALLOWED_DATASETS), required=True,
                        help="Destino: 'synthetic' o 'demo'.")
    parser.add_argument("--n", type=int, required=True,
                        help="Numero de registros a generar.")
    parser.add_argument("--seed", type=int, required=True,
                        help="Semilla para reproducibilidad.")
    args = parser.parse_args()

    print(f"Generando dataset '{args.dataset}' con n={args.n}, seed={args.seed}...")
    df = generate_synthetic_dataset(args.n, args.seed)

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"{args.dataset}.csv"
    df.to_csv(out_path, index=False)

    print(f"Guardado: {out_path}")
    print_validation_stats(df)


if __name__ == "__main__":
    main()
