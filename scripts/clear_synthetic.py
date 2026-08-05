"""CLI para eliminar datasets sinteticos (nunca elimina 'real').

Uso:
    python scripts/clear_synthetic.py --dataset synthetic
    python scripts/clear_synthetic.py --dataset demo
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
ALLOWED_DATASETS = {"synthetic", "demo"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Elimina un dataset sintetico")
    parser.add_argument("--dataset", choices=sorted(ALLOWED_DATASETS), required=True,
                        help="Debe ser 'synthetic' o 'demo'. Nunca 'real'.")
    args = parser.parse_args()

    path = DATA_DIR / f"{args.dataset}.csv"
    if not path.exists():
        print(f"No existe: {path}")
        return

    path.unlink()
    print(f"Eliminado: {path}")


if __name__ == "__main__":
    main()
