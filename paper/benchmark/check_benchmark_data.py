"""
Chequeo de archivos de entrada para el benchmark FiuMD vs TorchMD
=================================================================

Verifica que esten todos los archivos que necesita benchmark_fiumd_vs_torchmd.py
para cada proteina, ANTES de mandar a correr media hora de GPU y descubrir a la
tercera molecula que falta un .xtc.

Uso (chequea todas las proteinas que encuentre en datos/):
    python benchmark/check_benchmark_data.py

Uso (proteinas puntuales, mismo formato que el benchmark):
    python benchmark/check_benchmark_data.py --molecules cln:chignolin trp:trpcage

Uso (ademas de existir, abrir los archivos y chequear que sean coherentes):
    python benchmark/check_benchmark_data.py --deep

Uso (generar los que falten y tengan generador registrado):
    python benchmark/check_benchmark_data.py --fix

Termina con codigo 0 si no falta nada obligatorio, 1 si falta algo. Sirve para
encadenar:
    python benchmark/check_benchmark_data.py --molecules cln:chignolin && \
        python benchmark/benchmark_fiumd_vs_torchmd.py --molecules cln:chignolin --validate


Como agregar la generacion de un archivo faltante
--------------------------------------------------
El script esta partido en tres registros independientes, para que agregar
capacidades sea agregar una funcion y nada mas:

  REQUERIDOS   que archivos hacen falta (lista declarativa, abajo)
  GENERADORES  como fabricar cada uno    -> decorador @generador("<rol>")
  VALIDADORES  como chequear que sirva   -> decorador @validador("<rol>")

Para que --fix sepa construir, por ejemplo, el .xtc de clustering, alcanza con:

    @generador("xtc_clustering")
    def generar_xtc(ctx: Contexto) -> None:
        # ctx.ruta("xtc_clustering") es el destino
        # ctx.ruta("coords_5120"), ctx.mol, ctx.nombre_largo, etc. son la entrada
        ...

El decorador lo engancha solo: no hay que tocar el resto del script. Si un rol
no tiene generador, --fix lo dice explicitamente y sigue con los demas en vez de
cortar. `generar_model_modified` mas abajo es el ejemplo funcionando.
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Este script vive en benchmark/, pero datos/ cuelga de la raiz del repo.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Que archivos hacen falta
#
# Los patrones usan {mol} (nombre de la carpeta, ej "cln") y {largo} (nombre
# largo, ej "chignolin"). Es la misma convencion que documenta el docstring de
# benchmark_fiumd_vs_torchmd.py; si alla cambia, cambiar aca.
# ---------------------------------------------------------------------------
@dataclass
class ArchivoRequerido:
    rol: str
    patron: str
    descripcion: str
    origen: str = ""          # de donde sale, para el mensaje cuando falta
    obligatorio: bool = True  # False = lo usa el pipeline pero el benchmark no


REQUERIDOS: list[ArchivoRequerido] = [
    ArchivoRequerido(
        rol="forcefield",
        patron="{mol}/{mol}_prior_force_field.yaml",
        descripcion="Parametros de los priors (Bonds, RepulsionCG, Dihedrals)",
        origen="sale del ajuste de priors sobre la trayectoria all-atom",
    ),
    ArchivoRequerido(
        rol="topologia",
        patron="{mol}/{mol}.psf",
        descripcion="Topologia CG: atomos, tipos y enlaces",
        origen="se genera del PDB coarse-grained",
    ),
    ArchivoRequerido(
        rol="xtc_clustering",
        patron="{mol}/{mol}_kcenters_32clusters_coords.xtc",
        descripcion="Frames iniciales (32 clusters k-centers)",
        origen="clustering k-centers sobre la trayectoria CG",
    ),
    ArchivoRequerido(
        rol="coords_5120",
        patron="{mol}/{largo}_5120_ca_coords.npy",
        descripcion="Pool de coordenadas CA para muestrear estados iniciales",
        origen="submuestreo de la trayectoria CG",
    ),
    ArchivoRequerido(
        rol="embeddings",
        patron="{mol}/{largo}_ca_embeddings.npy",
        descripcion="Embeddings por bead que consume el modelo de torchmd-net",
        origen="mapeo de tipos de residuo a enteros",
    ),
    ArchivoRequerido(
        rol="model_modified",
        patron="{mol}/model_modified.ckpt",
        descripcion="Checkpoint del NNP con hyper_parameters['aggr'] seteado",
        origen="se deriva de model.ckpt (hay generador: correr con --fix)",
    ),
    ArchivoRequerido(
        rol="model_original",
        patron="{mol}/model.ckpt",
        descripcion="Checkpoint original de torchmd-net",
        origen="lo entrena/descarga el pipeline de training",
        # El benchmark no lo abre, pero sin el no se puede regenerar el
        # modificado, asi que se reporta sin marcarlo como bloqueante.
        obligatorio=False,
    ),
]

REQUERIDOS_POR_ROL = {a.rol: a for a in REQUERIDOS}


# ---------------------------------------------------------------------------
# Contexto: todo lo que un generador o validador puede necesitar
# ---------------------------------------------------------------------------
@dataclass
class Contexto:
    mol: str
    nombre_largo: str
    datos_dir: Path
    _cache: dict = field(default_factory=dict)

    def ruta(self, rol: str) -> Path:
        """Path absoluto del archivo de ese rol para esta proteina."""
        patron = REQUERIDOS_POR_ROL[rol].patron
        return self.datos_dir / patron.format(mol=self.mol, largo=self.nombre_largo)

    def existe(self, rol: str) -> bool:
        return self.ruta(rol).exists()

    def molecula(self):
        """Molecule de moleculekit con la topologia y los frames del xtc.

        Se cachea porque varios validadores la necesitan y leerla no es gratis.
        """
        if "mol" not in self._cache:
            from moleculekit.molecule import Molecule

            m = Molecule(str(self.ruta("topologia")))
            if self.existe("xtc_clustering"):
                m.read(str(self.ruta("xtc_clustering")))
            self._cache["mol"] = m
        return self._cache["mol"]


# ---------------------------------------------------------------------------
# Registro de generadores (--fix)
# ---------------------------------------------------------------------------
GENERADORES: dict[str, Callable[[Contexto], None]] = {}


def generador(rol: str):
    """Registra una funcion como la forma de fabricar el archivo de ese rol."""
    def deco(fn):
        if rol not in REQUERIDOS_POR_ROL:
            raise KeyError(f"rol desconocido: {rol!r}")
        GENERADORES[rol] = fn
        return fn
    return deco


@generador("model_modified")
def generar_model_modified(ctx: Contexto) -> None:
    """model.ckpt -> model_modified.ckpt.

    Los checkpoints viejos de torchmd-net traen el nombre de la reduccion en
    'reduce_op', pero la version instalada lo lee de 'aggr'. Si no se copia,
    el modelo se arma con aggr=None y falla al construir la red. Es el mismo
    arreglo que hacen los notebooks a mano.
    """
    import torch

    origen = ctx.ruta("model_original")
    if not origen.exists():
        raise FileNotFoundError(
            f"hace falta {origen} para poder generar model_modified.ckpt"
        )

    ckpt = torch.load(origen, weights_only=False, map_location="cpu")
    hp = ckpt.get("hyper_parameters")
    if hp is None:
        raise ValueError(f"{origen} no tiene 'hyper_parameters'; no es un ckpt de torchmd-net")
    if "reduce_op" not in hp:
        raise ValueError(f"{origen} no tiene hyper_parameters['reduce_op']; nada que copiar")

    hp["aggr"] = hp["reduce_op"]
    torch.save(ckpt, ctx.ruta("model_modified"))


# ---------------------------------------------------------------------------
# Registro de validadores (--deep)
#
# Cada uno recibe el contexto y devuelve un texto describiendo lo que encontro.
# Si algo no cierra, levanta una excepcion: el caller la convierte en error.
# ---------------------------------------------------------------------------
VALIDADORES: dict[str, Callable[[Contexto], str]] = {}


def validador(rol: str):
    def deco(fn):
        if rol not in REQUERIDOS_POR_ROL:
            raise KeyError(f"rol desconocido: {rol!r}")
        VALIDADORES[rol] = fn
        return fn
    return deco


@validador("topologia")
def validar_topologia(ctx: Contexto) -> str:
    import numpy as np

    mol = ctx.molecula()
    if mol.numAtoms == 0:
        raise ValueError("la topologia no tiene atomos")

    # Las masas son el campo silencioso del psf. Los .psf de
    # torchmd-protein-thermodynamics vienen con masa 0.0 en todos los beads y
    # hay que rellenarlos; si no, torchmd.parameters.Parameters igual los toma
    # (prefiere mol.masses sobre el yaml, y un array de ceros no es None), la
    # velocidad inicial sale sqrt(kT/0) = inf y la dinamica es NaN desde el
    # primer paso. El benchmark corre igual y da tiempos: hay que chequearlo aca.
    masas = np.asarray(mol.masses, dtype=float)
    if not np.isfinite(masas).all():
        raise ValueError("hay masas no finitas (NaN/inf) en el psf")
    nulas = int((masas <= 0).sum())
    if nulas:
        raise ValueError(
            f"{nulas}/{mol.numAtoms} beads con masa <= 0 en el psf. "
            f"La dinamica sale NaN. Rellenar la columna de masa con lo que "
            f"declara la seccion 'masses' del prior_force_field.yaml"
        )

    return (f"{mol.numAtoms} beads, {len(mol.bonds)} enlaces, "
            f"masas {masas.min():g}-{masas.max():g}")


@validador("xtc_clustering")
def validar_xtc(ctx: Contexto) -> str:
    mol = ctx.molecula()
    nframes = mol.coords.shape[2]
    if nframes == 0:
        raise ValueError("el xtc no tiene frames")
    if mol.coords.shape[0] != mol.numAtoms:
        raise ValueError(
            f"el xtc tiene {mol.coords.shape[0]} atomos y la topologia {mol.numAtoms}"
        )
    return f"{nframes} frames de {mol.coords.shape[0]} beads"


@validador("embeddings")
def validar_embeddings(ctx: Contexto) -> str:
    import numpy as np

    emb = np.load(ctx.ruta("embeddings"))
    if emb.ndim != 1:
        raise ValueError(f"se esperaba un vector 1D, vino shape {emb.shape}")

    # Estos .npy vienen guardados como strings ('4', '13', ...), no como enteros.
    # El benchmark los castea con .astype(int); aca se hace igual, si no el
    # chequeo de rango compara texto y da cualquier cosa.
    try:
        valores = emb.astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError(f"no se pueden leer como enteros (dtype {emb.dtype}): {e}") from e

    if valores.min() < 0:
        raise ValueError(f"hay embeddings negativos (min {valores.min()})")

    if ctx.existe("topologia"):
        natoms = ctx.molecula().numAtoms
        if len(valores) != natoms:
            raise ValueError(
                f"{len(valores)} embeddings para {natoms} beads de la topologia"
            )
    return (f"{len(valores)} embeddings, valores {valores.min()}..{valores.max()} "
            f"(guardados como {emb.dtype})")


@validador("coords_5120")
def validar_coords(ctx: Contexto) -> str:
    import numpy as np

    coords = np.load(ctx.ruta("coords_5120"))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"se esperaba (N, 3), vino shape {coords.shape}")
    detalle = f"{coords.shape[0]} posiciones"
    if ctx.existe("topologia"):
        natoms = ctx.molecula().numAtoms
        if coords.shape[0] % natoms:
            raise ValueError(
                f"{coords.shape[0]} posiciones no es multiplo de {natoms} beads"
            )
        detalle += f" = {coords.shape[0] // natoms} conformaciones de {natoms} beads"
    return detalle


@validador("forcefield")
def validar_forcefield(ctx: Contexto) -> str:
    import yaml

    with open(ctx.ruta("forcefield")) as f:
        ff = yaml.safe_load(f)
    if not isinstance(ff, dict):
        raise ValueError("el YAML no es un mapping")
    # El benchmark pide estos tres terminos; si no estan, revienta al construir.
    faltan = [k for k in ("bonds", "dihedrals") if k not in ff]
    if faltan:
        raise ValueError(f"al force field le faltan los terminos: {', '.join(faltan)}")
    return f"secciones: {', '.join(sorted(ff))}"


@validador("model_modified")
def validar_model_modified(ctx: Contexto) -> str:
    import torch

    ckpt = torch.load(ctx.ruta("model_modified"), weights_only=False, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    # Este es el chequeo que justifica que exista el archivo "modified": si aggr
    # quedo en None, el ckpt esta presente pero el modelo no se puede construir.
    if hp.get("aggr") is None:
        raise ValueError(
            "hyper_parameters['aggr'] es None: el ckpt no fue modificado "
            "(regenerarlo con --fix)"
        )
    return f"aggr={hp['aggr']!r}, modelo={hp.get('model', '?')}"


# ---------------------------------------------------------------------------
# Chequeo
# ---------------------------------------------------------------------------
def parse_molecule_spec(spec: str) -> tuple[str, str]:
    """'cln:chignolin' -> ('cln', 'chignolin'). Sin ':', se usa el mismo nombre.

    Misma funcion que en benchmark_fiumd_vs_torchmd.py, duplicada a proposito:
    este script tiene que poder correr sin importar el benchmark (que arrastra
    torch, moleculekit y fiumd solo para leer un argumento).
    """
    if ":" in spec:
        nombre, largo = spec.split(":", 1)
    else:
        nombre = largo = spec
    return nombre, largo


def descubrir_moleculas(datos_dir: Path) -> list[str]:
    """Arma las specs mirando que hay en datos/.

    El nombre largo se deduce del .npy de embeddings, que es el unico archivo
    que lo lleva siempre: <largo>_ca_embeddings.npy.
    """
    specs = []
    for sub in sorted(p for p in datos_dir.iterdir() if p.is_dir()):
        candidatos = sorted(sub.glob("*_ca_embeddings.npy"))
        largo = candidatos[0].name[: -len("_ca_embeddings.npy")] if candidatos else sub.name
        specs.append(f"{sub.name}:{largo}")
    return specs


def chequear_molecula(ctx: Contexto, deep: bool = False) -> list[dict]:
    """Una fila por archivo requerido."""
    filas = []
    for req in REQUERIDOS:
        ruta = ctx.ruta(req.rol)
        existe = ruta.exists()
        fila = dict(
            molecula=ctx.mol, rol=req.rol, ruta=str(ruta), descripcion=req.descripcion,
            origen=req.origen, obligatorio=req.obligatorio, existe=existe,
            tamano_bytes=ruta.stat().st_size if existe else 0,
            detalle="", error="",
        )

        if existe and deep and req.rol in VALIDADORES:
            try:
                fila["detalle"] = VALIDADORES[req.rol](ctx)
            except Exception as e:  # noqa: BLE001 - se reporta, no se propaga
                fila["error"] = f"{type(e).__name__}: {e}"

        filas.append(fila)
    return filas


def _humano(n: int) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if n < 1024 or unidad == "GB":
            return f"{n:.0f}{unidad}" if unidad == "B" else f"{n:.1f}{unidad}"
        n /= 1024.0
    return f"{n:.1f}GB"


def imprimir_molecula(mol: str, largo: str, filas: list[dict], deep: bool) -> None:
    print(f"\n=== {mol} ({largo}) ===")
    for f in filas:
        nombre = Path(f["ruta"]).name
        if not f["existe"]:
            marca = "FALTA " if f["obligatorio"] else "falta?"
            extra = f"  <- {f['origen']}" if f["origen"] else ""
            gen = "  [--fix lo genera]" if f["rol"] in GENERADORES else ""
            print(f"  {marca} {nombre}{extra}{gen}")
        elif f["error"]:
            print(f"  ROTO   {nombre}  ({_humano(f['tamano_bytes'])})")
            print(f"         {f['error']}")
        else:
            detalle = f"  {f['detalle']}" if deep and f["detalle"] else ""
            print(f"  OK     {nombre}  ({_humano(f['tamano_bytes'])}){detalle}")


def aplicar_fix(ctx: Contexto, filas: list[dict], dry_run: bool = False) -> list[dict]:
    """Corre los generadores registrados para lo que falte. Devuelve las filas
    re-chequeadas."""
    for f in filas:
        if f["existe"]:
            continue
        rol = f["rol"]
        nombre = Path(f["ruta"]).name
        if rol not in GENERADORES:
            print(f"  sin generador para {nombre} "
                  f"(agregar una funcion con @generador({rol!r}) en este script)")
            continue
        if dry_run:
            print(f"  [dry-run] generaria {nombre} con {GENERADORES[rol].__name__}()")
            continue
        try:
            print(f"  generando {nombre} ...", end=" ", flush=True)
            GENERADORES[rol](ctx)
            print("listo" if ctx.existe(rol) else "el generador no dejo el archivo")
        except Exception as e:  # noqa: BLE001 - se reporta y se sigue
            print(f"fallo: {type(e).__name__}: {e}")

    return chequear_molecula(ctx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--molecules", nargs="+", default=None,
                   help="specs 'carpeta:nombre_largo' (ej: cln:chignolin). "
                        "Por defecto se descubren solas mirando --datos-dir")
    p.add_argument("--datos-dir", default=str(REPO_ROOT / "datos"))
    p.add_argument("--deep", action="store_true",
                   help="ademas de que existan, abrirlos y chequear que sean "
                        "coherentes entre si (necesita numpy, torch y moleculekit)")
    p.add_argument("--fix", action="store_true",
                   help="genera los archivos faltantes que tengan generador registrado")
    p.add_argument("--dry-run", action="store_true",
                   help="con --fix, muestra que generaria sin escribir nada")
    p.add_argument("--json", metavar="ARCHIVO", default=None,
                   help="ademas del reporte, escribe el resultado completo a un JSON")
    p.add_argument("--list-builders", action="store_true",
                   help="lista los roles que --fix sabe generar y termina")
    args = p.parse_args()

    if args.list_builders:
        print("Roles con generador registrado (--fix):")
        for rol in REQUERIDOS_POR_ROL:
            fn = GENERADORES.get(rol)
            estado = f"{fn.__name__}()" if fn else "-- sin generador --"
            print(f"  {rol:<16} {estado}")
        print("\nPara agregar uno, definir en este script:\n"
              "    @generador(\"<rol>\")\n"
              "    def generar_<algo>(ctx: Contexto) -> None: ...")
        return 0

    datos_dir = Path(args.datos_dir)
    if not datos_dir.is_dir():
        print(f"No existe el directorio de datos: {datos_dir}", file=sys.stderr)
        return 1

    specs = args.molecules if args.molecules else descubrir_moleculas(datos_dir)
    if not specs:
        print(f"No encontre ninguna proteina en {datos_dir}", file=sys.stderr)
        return 1
    if not args.molecules:
        print(f"Proteinas encontradas en {datos_dir}: {', '.join(specs)}")

    todas, listas, incompletas = [], [], []
    for spec in specs:
        mol, largo = parse_molecule_spec(spec)
        ctx = Contexto(mol=mol, nombre_largo=largo, datos_dir=datos_dir)

        filas = chequear_molecula(ctx, deep=args.deep)
        if args.fix and any(not f["existe"] for f in filas):
            print(f"\n=== {mol} ({largo}) — generando faltantes ===")
            filas = aplicar_fix(ctx, filas, dry_run=args.dry_run)
            if args.deep:
                filas = chequear_molecula(ctx, deep=True)

        imprimir_molecula(mol, largo, filas, args.deep)
        todas.extend(filas)

        problemas = [f for f in filas
                     if f["obligatorio"] and (not f["existe"] or f["error"])]
        (incompletas if problemas else listas).append((spec, problemas))

    print(f"\n{'=' * 62}")
    if listas:
        print(f"LISTAS para el benchmark ({len(listas)}): "
              f"{', '.join(s for s, _ in listas)}")
    for spec, problemas in incompletas:
        faltan = [Path(f["ruta"]).name for f in problemas if not f["existe"]]
        rotos = [Path(f["ruta"]).name for f in problemas if f["error"]]
        print(f"INCOMPLETA {spec}:")
        if faltan:
            print(f"    faltan: {', '.join(faltan)}")
        if rotos:
            print(f"    rotos:  {', '.join(rotos)}")
    if not incompletas:
        print("No falta nada. Se puede correr el benchmark con:")
        print(f"    python benchmark/benchmark_fiumd_vs_torchmd.py --molecules {' '.join(specs)} "
              f"--validate")
    print("=" * 62)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(datos_dir=str(datos_dir), deep=args.deep, archivos=todas),
                      f, indent=2)
        print(f"Reporte JSON en {args.json}")

    return 1 if incompletas else 0


if __name__ == "__main__":
    sys.exit(main())
