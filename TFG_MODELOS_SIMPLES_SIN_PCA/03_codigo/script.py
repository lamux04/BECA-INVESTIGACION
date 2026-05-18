from pathlib import Path
import json
import re
import shutil

RUTA_BASE = Path(".")
HACER_BACKUP = True

NOTEBOOKS = list(RUTA_BASE.rglob("*.ipynb"))

print(f"Notebooks encontrados: {len(NOTEBOOKS)}")

for nb_path in NOTEBOOKS:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modificado = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        original = source

        # 1. Quitar import de PCA
        source = re.sub(
            r"^\s*from sklearn\.decomposition import PCA\s*\n",
            "",
            source,
            flags=re.MULTILINE
        )

        # 2. Quitar configuración de PCA
        source = re.sub(
            r"\n?# ===== CONFIG PCA =====\nN_COMPONENTS_PCA\s*=\s*\d+\s*\n?",
            "\n",
            source
        )

        source = re.sub(
            r"^\s*N_COMPONENTS_PCA\s*=\s*\d+\s*\n",
            "",
            source,
            flags=re.MULTILINE
        )

        # 3. Quitar paso PCA del pipeline
        source = re.sub(
            r'\n\s*\("pca",\s*PCA\(n_components=N_COMPONENTS_PCA\)\),',
            "",
            source
        )

        source = re.sub(
            r'\n\s*\("pca",\s*PCA\(n_components\s*=\s*N_COMPONENTS_PCA\)\),',
            "",
            source
        )

        # 4. Quitar n_components_pca de summaries/diccionarios
        source = re.sub(
            r'\n\s*"n_components_pca"\s*:\s*N_COMPONENTS_PCA,?',
            "",
            source
        )

        # 5. Añadir use_pca=False si quieres dejar constancia
        if '"parametros"' in source and '"use_pca"' not in source:
            source = re.sub(
                r'("p"\s*:\s*P,?)',
                r'\1\n        "use_pca": False,',
                source
            )

        # 6. Quitar pca4_ del nombre del experimento
        source = re.sub(r"pca\d+_", "", source, flags=re.IGNORECASE)
        source = re.sub(r"pca_\d+_", "", source, flags=re.IGNORECASE)

        if source != original:
            cell["source"] = source.splitlines(keepends=True)
            modificado = True

    if modificado:
        if HACER_BACKUP:
            backup_path = nb_path.with_suffix(".ipynb.bak")
            shutil.copy2(nb_path, backup_path)

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        print(f"Modificado: {nb_path}")
    else:
        print(f"Sin cambios: {nb_path}")

print("Proceso terminado.")