"""Analisis de RMSD y MSD de las cuatro proteinas en una sola pasada.

Extiende fig_rmsd.py al caso multi-proteina. Lo que cambia respecto de mirar
una sola no es el calculo sino la escala: las cuatro corridas juntas son 8.7 GB
de coordenadas y las trayectorias van de 1.56 ns (cln) a 25 ns (lambda). Eso
fuerza tres decisiones:

1. Los resultados derivados se cachean en disco: 150 MB las cuatro juntas
   contra 8.7 GB de coordenadas. Una vez calculados, el notebook re-dibuja
   figuras en segundos sin volver a tocar los .npy.

2. El MSD se calcula por FFT. MDutils.metrics.MSD recorre un lag por frame y es
   O(N^2): sobre lambda, con 249900 frames, son minutos; por FFT es 1.1 s. Los
   dos dan el mismo numero (verificado, difieren 6e-7, que es el float32 de
   MDutils), y `metodo="mdutils"` sigue disponible.

3. El centro de masa se acumula por bloques de frames. El array de lambda son
   1.92 GB y el transpose lo duplica; por bloques el pico queda en 2 GB.

Las figuras son tres compuestas y no doce sueltas: un panel por proteina para
RMSD y distribucion, y una sola grafica normalizada para el MSD.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp

from fig_rmsd import (
    ANCHO, COLOR_FIUMD, COLOR_TORCHMD, COLOR_GRID, COLOR_MUTED, GUIONES, MM,
    DPI, _ejes, _limites_densidad, difusion_libre, estilo, referencia_ca,
    rmsd_motor,
)

# Color por proteina para la figura de MSD, donde el motor se distingue por el
# trazo y no por el color. Los dos primeros son los de FiuMD y TorchMD, que ahi
# ya no cumplen esa funcion.
COLOR_PROTEINA = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#8a63d2"}


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
def centro_de_masa(coords_dir, prefijo, motor, masses, bloque=20000):
    """(nreplicas, nframes, 3) acumulado por bloques de frames.

    Nunca se trae el array entero a RAM: en lambda son 1.92 GB y el COM que
    queda son 6 MB.
    """
    arr = np.load(Path(coords_dir) / f"{prefijo}_{motor}.npy", mmap_mode="r")
    w = np.asarray(masses, dtype=np.float64)
    partes = [
        np.einsum("frad,a->frd", np.asarray(arr[j:j + bloque], dtype=np.float64), w) / w.sum()
        for j in range(0, arr.shape[0], bloque)
    ]
    return np.concatenate(partes).transpose(1, 0, 2)


def msd_fft(com):
    """MSD promediado sobre origenes de tiempo: (R, N, 3) -> (R, N).

    Algoritmo de correlacion rapida: el promedio sobre origenes es una
    autocorrelacion, y por FFT sale en O(N log N) en vez de O(N^2).
    """
    r = np.asarray(com, dtype=np.float64)
    R, N, _ = r.shape
    D = (r ** 2).sum(-1)
    D = np.concatenate([D, np.zeros((R, 1))], axis=1)
    n = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(r, n=n, axis=1)
    S2 = np.fft.irfft(F * F.conjugate(), n=n, axis=1)[:, :N].sum(-1) / (N - np.arange(N))
    Q = 2 * D.sum(1)
    S1 = np.zeros((R, N))
    for k in range(N):
        Q = Q - D[:, k - 1] - D[:, N - k]
        S1[:, k] = Q / (N - k)
    return S1 - 2 * S2


def msd_motor(coords_dir, prefijo, motor, masses, device, metodo="fft", bloque=20000):
    """MSD del COM. `metodo` "fft" o "mdutils"; dan el mismo numero."""
    com = centro_de_masa(coords_dir, prefijo, motor, masses, bloque)
    if metodo == "fft":
        return msd_fft(com)
    import MDutils
    # Se le pasa el COM como una trayectoria de un solo atomo de masa 1: es su
    # propio centro de masa, asi que MDutils lo deja igual.
    return MDutils.metrics.MSD(
        com.astype(np.float32)[:, :, None, :], np.ones(1), device)


def calcular(cfg, device, cache="results_validacion_larga/analisis",
             temp=350.0, gamma_ps=1.0, dt_fs=1.0, metodo="fft", rehacer=False):
    """Calcula (o lee del cache) RMSD y MSD de una proteina.

    `cfg` es un dict con nombre, coords, prefijo, referencia y psf. Devuelve un
    dict con los arrays derivados, que es lo unico que necesitan las figuras.
    """
    from moleculekit.molecule import Molecule

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / f"{cfg['prefijo']}.npz"

    masses = Molecule(cfg["psf"]).masses
    difusion = difusion_libre(masses, temp, gamma_ps)
    pasos = np.load(Path(cfg["coords"]) / f"{cfg['prefijo']}_pasos.npy")
    # El stride sale del archivo de pasos y no de una constante a mano: un valor
    # desfasado del real corre todo el eje temporal sin avisar.
    stride = int(pasos[1] - pasos[0])

    if npz.exists() and not rehacer:
        d = dict(np.load(npz))
    else:
        ref = referencia_ca(cfg["referencia"])
        d = {
            "rmsd_f": rmsd_motor(cfg["coords"], cfg["prefijo"], "fiumd", ref, device),
            "rmsd_t": rmsd_motor(cfg["coords"], cfg["prefijo"], "torchmd", ref, device),
            "msd_f": msd_motor(cfg["coords"], cfg["prefijo"], "fiumd", masses, device, metodo),
            "msd_t": msd_motor(cfg["coords"], cfg["prefijo"], "torchmd", masses, device, metodo),
        }
        np.savez_compressed(npz, **d)

    d["nombre"] = cfg["nombre"]
    d["t_ns"] = pasos * dt_fs / 1e6
    d["tau_ps"] = np.arange(d["msd_f"].shape[1]) * stride * dt_fs / 1000.0
    d["difusion"] = difusion
    d["masa"] = float(np.sum(masses))
    return d


def tabla_resumen(resultados):
    """Una fila (dict) por proteina. Es el resultado principal; va como tabla,
    no como figura.

    Devuelve una lista de dicts y no un DataFrame para no arrastrar pandas.
    En el notebook, `pd.DataFrame(tabla_resumen(res))` la muestra formateada;
    `tabla_texto` hace lo mismo sin pandas.
    """
    filas = []
    for d in resultados:
        paso = max(1, d["rmsd_f"].size // 50000)
        ks = ks_2samp(d["rmsd_f"].ravel()[::paso], d["rmsd_t"].ravel()[::paso]).statistic
        k = len(d["tau_ps"]) // 10 - 1       # el lag mas largo que se grafica
        cociente = d["msd_f"][:, k].mean() / (6 * d["difusion"] * d["tau_ps"][k])
        filas.append({
            "proteina": d["nombre"],
            "replicas": d["rmsd_f"].shape[0],
            "duracion (ns)": round(float(d["t_ns"][-1]), 2),
            "masa (amu)": d["masa"],
            "RMSD FiuMD": f"{d['rmsd_f'].mean():.2f} ± {d['rmsd_f'].std():.2f}",
            "RMSD TorchMD": f"{d['rmsd_t'].mean():.2f} ± {d['rmsd_t'].std():.2f}",
            "dif. medias": round(float(d["rmsd_f"].mean() - d["rmsd_t"].mean()), 3),
            "D_KS": round(float(ks), 4),
            "D (Å²/ps)": round(float(d["difusion"]), 3),
            "MSD/6Dτ": round(float(cociente), 3),
        })
    return filas


def tabla_texto(filas):
    """La misma tabla alineada en texto plano, sin depender de pandas."""
    cols = list(filas[0])
    anchos = [max(len(c), *(len(str(f[c])) for f in filas)) for c in cols]
    lineas = ["  ".join(c.rjust(a) for c, a in zip(cols, anchos)),
              "  ".join("-" * a for a in anchos)]
    lineas += ["  ".join(str(f[c]).rjust(a) for c, a in zip(cols, anchos))
               for f in filas]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _mostrar(fig, salida):
    """Guarda y devuelve la figura en vez de cerrarla.

    En un notebook, devolver el Figure hace que la celda la muestre; si se
    cerrara como en fig_rmsd.py (pensado para linea de comandos) la celda solo
    imprimiria una ruta.
    """
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    ruta = salida.with_suffix(".png")
    fig.savefig(ruta, dpi=DPI)
    print(ruta)
    return fig


def _letra(ax, texto):
    ax.text(0.0, 1.04, texto, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def _grilla(n=4, alto=55):
    """Lienzo 2x2 al ancho del documento."""
    estilo()
    fig, axes = plt.subplots(2, 2, figsize=(ANCHO, 2 * alto * MM),
                             layout="constrained")
    return fig, axes.ravel()[:n]


def figura_paneles_rmsd(resultados, salida):
    """Trazas de RMSD, un panel por proteina.

    Ejes independientes a proposito: las trayectorias van de 1.56 a 25 ns y los
    RMSD de cada proteina viven en rangos distintos. Con ejes compartidos, tres
    de los cuatro paneles quedan aplastados.
    """
    import random

    fig, axes = _grilla(len(resultados))
    for ax, d, letra in zip(axes, resultados, "abcd"):
        trazas = ([(COLOR_FIUMD, r) for r in d["rmsd_f"]] +
                  [(COLOR_TORCHMD, r) for r in d["rmsd_t"]])
        # Orden mezclado: dibujar un motor entero y despues el otro deja al
        # segundo sistematicamente arriba, y parece mas denso.
        random.Random(0).shuffle(trazas)
        for color, r in trazas:
            ax.plot(d["t_ns"], r, lw=0.12, alpha=0.18, color=color, rasterized=True)
        ax.plot(d["t_ns"], d["rmsd_f"].mean(0), lw=1.0, color=COLOR_FIUMD, zorder=3)
        ax.plot(d["t_ns"], d["rmsd_t"].mean(0), lw=1.0, color=COLOR_TORCHMD,
                ls=GUIONES, zorder=4)
        _ejes(ax)
        ax.set_xlim(d["t_ns"][0], d["t_ns"][-1])
        ax.set_title(f"{d['nombre']}  ({d['rmsd_f'].shape[0]} réplicas)", pad=3)
        _letra(ax, letra)
    for ax in axes[2:]:
        ax.set_xlabel("tiempo (ns)")
    for ax in axes[::2]:
        ax.set_ylabel(r"RMSD C$\alpha$ ($\mathrm{\AA}$)")
    axes[0].legend(handles=[
        Line2D([], [], color=COLOR_FIUMD, lw=1.2, label="FiuMD"),
        Line2D([], [], color=COLOR_TORCHMD, lw=1.2, ls=GUIONES, label="TorchMD")],
        loc="upper right", ncol=2, handlelength=1.4, columnspacing=1.0,
        borderaxespad=0.3)
    return _mostrar(fig, salida)


def figura_paneles_distribucion(resultados, salida, ancho_bin=None):
    """Densidad del RMSD, un panel por proteina.

    `ancho_bin` en None escala con el tamano de muestra (regla de Scott sobre
    el desvio): 0.1 A es razonable para los 2 millones de frames de trp, pero
    deja dentada una corrida corta.
    """
    fig, axes = _grilla(len(resultados))
    for ax, d, letra in zip(axes, resultados, "abcd"):
        rf, rt = d["rmsd_f"], d["rmsd_t"]
        ancho = ancho_bin or max(0.02, round(
            3.5 * min(rf.std(), rt.std()) / rf.size ** (1 / 3), 2))
        lo = min(rf.min(), rt.min())
        hi = max(rf.max(), rt.max())
        bins = np.arange(np.floor(lo * 10) / 10, hi + ancho, ancho)

        # Uno relleno y el otro como linea encima: dos rellenos se tapan, dos
        # lineas del mismo trazo se confunden.
        dens = []
        for r, color, kw in (
                (rf, COLOR_FIUMD, dict(histtype="stepfilled", alpha=0.22, lw=0, zorder=2)),
                (rf, COLOR_FIUMD, dict(histtype="step", lw=0.9, zorder=3)),
                (rt, COLOR_TORCHMD, dict(histtype="step", lw=0.8, ls=GUIONES, zorder=4))):
            n, _, _ = ax.hist(r.ravel(), bins=bins, density=True, color=color, **kw)
            dens.append(n)

        xlim, ylim = _limites_densidad(rf, rt, dens)
        _ejes(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{d['nombre']}  (bin {ancho:g} " r"$\mathrm{\AA}$)", pad=3)
        _letra(ax, letra)
    for ax in axes[2:]:
        ax.set_xlabel(r"RMSD C$\alpha$ ($\mathrm{\AA}$)")
    for ax in axes[::2]:
        ax.set_ylabel(r"densidad ($\mathrm{\AA}^{-1}$)")
    axes[0].legend(handles=[
        Patch(facecolor=COLOR_FIUMD, alpha=0.22, edgecolor=COLOR_FIUMD, lw=0.9,
              label="FiuMD"),
        Line2D([], [], color=COLOR_TORCHMD, lw=1.2, ls=GUIONES, label="TorchMD")],
        loc="upper right", handlelength=1.4, borderaxespad=0.3)
    return _mostrar(fig, salida)


def figura_msd_normalizado(resultados, salida, alto=80):
    """MSD / 6D(tau) de las cuatro proteinas en una sola grafica.

    Normalizado y no crudo: cada proteina tiene su masa, o sea su propio D (960
    amu en lambda contra 120 en cln), asi que el MSD crudo son cuatro rectas
    paralelas que no dicen nada. Dividido por la prediccion de particula libre,
    las ocho curvas colapsan sobre 1 y se ve de un golpe que los dos motores
    reproducen la difusion libre en las cuatro proteinas.

    El arranque por debajo de 1 no es un error: hasta tiempos del orden de
    1/gamma = 1 ps el movimiento es balistico, no difusivo.
    """
    estilo()
    fig, ax = plt.subplots(figsize=(ANCHO, alto * MM), layout="constrained")

    for i, d in enumerate(resultados):
        s = slice(1, len(d["tau_ps"]) // 10)
        tau = d["tau_ps"][s]
        libre = 6 * d["difusion"] * tau
        color = COLOR_PROTEINA[i % len(COLOR_PROTEINA)]
        ax.plot(tau, d["msd_f"].mean(0)[s] / libre, lw=1.1, color=color)
        ax.plot(tau, d["msd_t"].mean(0)[s] / libre, lw=1.1, color=color, ls=GUIONES)

    ax.axhline(1.0, color=COLOR_MUTED, lw=0.7)
    ax.set_xscale("log")
    _ejes(ax, which="both")
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"MSD $/\ 6D\tau$")
    ax.set_ylim(0, 1.15)

    # Dos leyendas: el color dice la proteina y el trazo dice el motor. En una
    # sola entrada por curva serian ocho renglones para dos atributos.
    leyenda_prot = ax.legend(handles=[
        Line2D([], [], color=COLOR_PROTEINA[i % len(COLOR_PROTEINA)], lw=1.2,
               label=d["nombre"])
        for i, d in enumerate(resultados)],
        loc="lower right", ncol=len(resultados), handlelength=1.4,
        columnspacing=1.0, borderaxespad=0.5)
    ax.add_artist(leyenda_prot)
    ax.legend(handles=[
        Line2D([], [], color=COLOR_MUTED, lw=1.2, label="FiuMD"),
        Line2D([], [], color=COLOR_MUTED, lw=1.2, ls=GUIONES, label="TorchMD")],
        loc="upper left", handlelength=1.6, borderaxespad=0.5)
    return _mostrar(fig, salida)
