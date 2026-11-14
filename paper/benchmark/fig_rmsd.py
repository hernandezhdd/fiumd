"""Las tres figuras de RMSD.ipynb, cada una en su propio PNG.

Reproduce lo que quedo en el notebook: trazas de RMSD, distribucion de
probabilidad del RMSD y MSD del centro de masa, comparando FiuMD con TorchMD.

El ancho esta fijado en 170 mm, que es el `total` que declara el \\geometry de
reporte_resultados/main.tex. Generando la figura con ese ancho exacto,
\\includegraphics[width=\\linewidth] no la reescala y los cuerpos de letra que se
piden aca son los que terminan impresos. Por eso tampoco se usa
bbox_inches="tight": recorta el lienzo a lo que ocupan los datos, el archivo
sale mas angosto y LaTeX despues lo estira. El espaciado lo resuelve
constrained_layout, que acomoda dentro del lienzo en vez de recortarlo.

Salida en PNG a 600 dpi. Si en algun momento hace falta el vectorial para el
LaTeX, alcanza con agregar la linea del .pdf en `_guardar`: las trayectorias ya
van rasterizadas, que es lo que evita un PDF de decenas de MB.

Uso:
    python benchmark/fig_rmsd.py \\
        --coords results_validacion_larga/cln/coords --prefijo cln_b128 \\
        --referencia datos/experimental_structures/chignolin_cln025.pdb \\
        --psf datos/cln/cln.psf --salida reporte_resultados/figuras

    python benchmark/fig_rmsd.py \\
        --coords results_validacion_larga/bbl/coords --prefijo bbl_b16 \\
        --referencia datos/experimental_structures/bbl_2wxc.pdb \\
        --psf datos/bbl/bbl.psf --salida reporte_resultados/figuras


    python benchmark/fig_rmsd.py --coords results_validacion_larga/lambda/coords --prefijo lambda_b8    --referencia datos/experimental_structures/lambda_1lmb.pdb --psf datos/lambda/lambda.psf --salida reporte_resultados/figuras

        

"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp

# Paleta compartida con plot_benchmark_results.py: las figuras del informe
# tienen que hablar el mismo idioma de color.
COLOR_FIUMD = "#2a78d6"
COLOR_TORCHMD = "#eb6834"
COLOR_GRID = "#b8b8b3"
COLOR_MUTED = "#52514e"

MM = 1 / 25.4
ANCHO = 170 * MM          # \linewidth del main.tex
GUIONES = (0, (4, 3))   # patron unico para TorchMD en las tres figuras
DPI = 300               # el minimo habitual de las editoriales para raster

KB = 0.001987191          # kcal/(mol K)
CONV = 418.4              # (A/ps)^2 por cada (kcal/mol)/amu


def estilo():
    """rcParams para figura de articulo: cuerpos chicos y lineas finas, porque
    la figura se imprime al tamano en que se genera."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def _ejes(ax, which="major"):
    """Grilla y recuadro recesivos: los datos adelante, el andamiaje atras."""
    ax.grid(alpha=0.3, color=COLOR_GRID, linewidth=0.5, which=which)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(COLOR_GRID)
        s.set_linewidth(0.8)
    ax.tick_params(colors=COLOR_MUTED)


def _guardar(fig, salida):
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida.with_suffix(".png"), dpi=DPI)
    plt.close(fig)
    return salida.with_suffix(".png")


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
def referencia_ca(pdb):
    """(1, 1, natoms, 3) con los CA de la estructura experimental."""
    from moleculekit.molecule import Molecule
    mol = Molecule(str(pdb))
    mol.filter("name CA")
    return torch.from_numpy(np.ascontiguousarray(mol.coords[:, :, 0]))[None, None]


def rmsd_motor(coords_dir, prefijo, motor, ref, device, chunk=8, chunk_frames=20000):
    """RMSD (nreplicas, nframes) contra `ref`, por tandas de replicas y frames.

    El archivo en disco es (frames, replicas, natoms, 3) y MDutils espera
    (replicas, frames, natoms, 3). Kabsch no trabaja in-place: entre el
    contiguous del permute y las copias internas, hacerlo de una sola vez pide
    varios GB de VRAM. Por tandas el tiempo es el mismo y la memoria escala con
    `chunk * chunk_frames`. Partir solo por replicas no alcanza cuando hay pocas
    replicas y muchos frames (lambda_b8: 8 x 250k frames = 1.9 GB por tanda).
    """
    import MDutils
    arr = np.load(Path(coords_dir) / f"{prefijo}_{motor}.npy", mmap_mode="r")
    nframes, nrep = arr.shape[:2]
    filas = []
    for i in range(0, nrep, chunk):
        filas.append(torch.cat([
            MDutils.metrics.RMSD(
                torch.from_numpy(np.array(arr[j:j + chunk_frames, i:i + chunk])).permute(1, 0, 2, 3).contiguous(),
                ref, device).cpu()
            for j in range(0, nframes, chunk_frames)
        ], dim=1))
    return torch.cat(filas).numpy()


def msd_motor(coords_dir, prefijo, motor, masses, device, chunk_frames=20000):
    """MSD del centro de masa (nreplicas, nlags), promediado sobre origenes.

    El transpose no es opcional: sin el, MDutils toma el indice de replica como
    si fuera el tiempo y mide el desplazamiento entre trayectorias distintas.
    No falla, devuelve un numero sin sentido.
    """
    import MDutils
    arr = np.load(Path(coords_dir) / f"{prefijo}_{motor}.npy", mmap_mode="r")
    # MDutils.MSD sube la trayectoria entera a la GPU y la duplica al pesar por
    # masa. El COM se calcula aca en CPU por tandas y se le pasa como una
    # trayectoria de un solo atomo de masa 1, que es su propio centro de masa.
    w = np.asarray(masses, dtype=np.float64)
    com = np.concatenate([
        np.einsum("frad,a->frd", np.asarray(arr[j:j + chunk_frames], dtype=np.float64), w) / w.sum()
        for j in range(0, arr.shape[0], chunk_frames)
    ]).astype(np.float32)
    return MDutils.metrics.MSD(com.transpose(1, 0, 2)[:, :, None, :], np.ones(1), device)


def difusion_libre(masses, temp, gamma_ps):
    """D = kB T / (gamma M) en A^2/ps.

    Sobre el centro de masa no actua ninguna fuerza externa: priors y NNP son
    internos y se cancelan al sumar. Asi que el COM tiene que difundir como una
    particula libre bajo Langevin, y la curva teorica no tiene parametros
    libres. Es un test directo del termostato.
    """
    return KB * temp / (gamma_ps * np.sum(masses)) * CONV


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_rmsd(t_ns, rmsd_f, rmsd_t, molecula, salida, ymin=1.8):
    """Trazas de RMSD de las dos corridas, con la media del ensemble.

    Media y no mediana para que la curva gruesa signifique lo mismo que en la
    figura de MSD: el lector no deberia tener que mirar el pie para saber que
    estadistico esta leyendo.
    """
    estilo()
    fig, ax = plt.subplots(figsize=(ANCHO, 62 * MM), layout="constrained")

    # Orden de dibujo mezclado: si primero van todas las de FiuMD y despues
    # todas las de TorchMD, el segundo motor queda sistematicamente arriba y
    # parece mas denso.
    trazas = ([(COLOR_FIUMD, r) for r in rmsd_f] +
              [(COLOR_TORCHMD, r) for r in rmsd_t])
    random.Random(0).shuffle(trazas)
    for color, r in trazas:
        ax.plot(t_ns, r, lw=0.15, alpha=0.20, color=color, rasterized=True)

    # Las medias se pisan casi exactamente: si las dos fueran solidas, la
    # segunda taparia a la primera y pareceria que hay una sola curva.
    ax.plot(t_ns, rmsd_f.mean(axis=0), lw=1.1, color=COLOR_FIUMD, zorder=3)
    ax.plot(t_ns, rmsd_t.mean(axis=0), lw=1.1, color=COLOR_TORCHMD,
            ls=GUIONES, zorder=4)

    _ejes(ax)
    ax.set_xlim(t_ns[0], t_ns[-1])
    ax.set_ylim(ymin, max(rmsd_f.max(), rmsd_t.max()) * 1.05)
    ax.set_xlabel("tiempo (ns)")
    ax.set_ylabel(r"RMSD C$\alpha$ ($\mathrm{\AA}$)")
    ax.set_title(f"RMSD - {molecula}")
    ax.legend(handles=[
        Line2D([], [], color=COLOR_FIUMD, lw=1.2, label="FiuMD"),
        Line2D([], [], color=COLOR_TORCHMD, lw=1.2, ls=GUIONES,
               label="TorchMD")],
        loc="upper right", ncol=2, handlelength=1.4, columnspacing=1.2,
        borderaxespad=0.3)
    return _guardar(fig, salida)


def _limites_densidad(rmsd_f, rmsd_t, densidades):
    """(xlim, ylim) deducidos del dato.

    Cablear los limites ata la figura a una proteina: el RMSD de cln vive cerca
    de 1-2 A y el de bbl por encima de 6, asi que un rango fijo para trp los
    recorta o los deja en un rincon. Se toman los percentiles 0.05 y 99.95 de
    los dos motores juntos —no el minimo y el maximo, que son un frame suelto y
    estiran el eje por nada— redondeados hacia afuera al medio angstrom.
    """
    juntos = np.concatenate([rmsd_f.ravel(), rmsd_t.ravel()])
    lo, hi = np.percentile(juntos, [0.05, 99.95])
    xlim = (np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2)
    ylim = (0.0, max(d.max() for d in densidades) * 1.12)
    return xlim, ylim


def figura_distribucion(rmsd_f, rmsd_t, molecula, salida,
                        ancho_bin=0.1, xlim=None, ylim=None, alto=80 * MM):
    """Densidad de probabilidad del RMSD de los dos motores.

    `ancho_bin` fijo y redondo, declarado en el pie de figura: es lo que vuelve
    reproducible un histograma. Freedman-Diaconis daria 0.012 A sobre estos 2
    millones de muestras, pero estan correlacionadas (stride de 100 fs contra
    tiempos de correlacion mucho mayores), asi que esa finura dibujaria ruido
    correlacionado y no estructura.

    `xlim` e `ylim` en None se deducen del dato; pasarlos los fuerza.
    """
    estilo()
    lo = min(rmsd_f.min(), rmsd_t.min())
    hi = max(rmsd_f.max(), rmsd_t.max())
    bins = np.arange(np.floor(lo * 10) / 10, hi + ancho_bin, ancho_bin)

    fig, ax = plt.subplots(figsize=(ANCHO, alto), layout="constrained")

    # Uno relleno y el otro como linea encima: dos rellenos se tapan, dos
    # lineas del mismo trazo se confunden.
    dens = []
    for r, color, kw in ((rmsd_f, COLOR_FIUMD, dict(histtype="stepfilled", alpha=0.22, lw=0, zorder=2)),
                         (rmsd_f, COLOR_FIUMD, dict(histtype="step", lw=1.0, zorder=3)),
                         (rmsd_t, COLOR_TORCHMD, dict(histtype="step", lw=0.9,
                                                      ls=GUIONES, zorder=4))):
        n, _, _ = ax.hist(r.ravel(), bins=bins, density=True, color=color, **kw)
        dens.append(n)

    auto_x, auto_y = _limites_densidad(rmsd_f, rmsd_t, dens)
    _ejes(ax)
    ax.set_xlim(*(xlim or auto_x))
    ax.set_ylim(*(ylim or auto_y))
    ax.set_xlabel(r"RMSD C$\alpha$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"densidad de probabilidad ($\mathrm{\AA}^{-1}$)")
    ax.set_title(f"Distribución de Probabilidad para {molecula}")
    ax.legend(handles=[
        Patch(facecolor=COLOR_FIUMD, alpha=0.22, edgecolor=COLOR_FIUMD, lw=1.0,
              label="FiuMD"),
        Line2D([], [], color=COLOR_TORCHMD, lw=1.2, ls=GUIONES,
               label="TorchMD")],
        loc="upper right", bbox_to_anchor=(1.0, 0.98), handlelength=1.4,
        borderaxespad=0.3)
    return _guardar(fig, salida)


def figura_msd(tau_ps, msd_f, msd_t, difusion, molecula, salida, alto=80 * MM):
    """MSD del centro de masa contra la prediccion de particula libre.

    Se grafica hasta el 10 % de los lags: el MSD a lag k promedia sobre N-k
    origenes de tiempo, asi que en el ultimo lag hay uno solo y el valor es
    ruido. Cortar en el 10 % de la trayectoria es la convencion.
    """
    estilo()
    s = slice(1, len(tau_ps) // 10)
    fig, ax = plt.subplots(figsize=(ANCHO, alto), layout="constrained")

    for m, color in ((msd_f, COLOR_FIUMD), (msd_t, COLOR_TORCHMD)):
        for r in m:
            ax.plot(tau_ps[s], r[s], lw=0.15, alpha=0.18, color=color,
                    rasterized=True)

    ax.plot(tau_ps[s], msd_f.mean(0)[s], lw=1.2, color=COLOR_FIUMD, zorder=3)
    ax.plot(tau_ps[s], msd_t.mean(0)[s], lw=1.2, color=COLOR_TORCHMD,
            ls=GUIONES, zorder=4)
    ax.plot(tau_ps[s], 6 * difusion * tau_ps[s], lw=1.0, color=COLOR_MUTED,
            zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _ejes(ax, which="both")
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"MSD del centro de masa ($\mathrm{\AA}^{2}$)")
    ax.set_title(f"MSD para {molecula}")
    ax.legend(handles=[
        Line2D([], [], color=COLOR_FIUMD, lw=1.2, label="FiuMD"),
        Line2D([], [], color=COLOR_TORCHMD, lw=1.2, ls=GUIONES,
               label="TorchMD"),
        Line2D([], [], color=COLOR_MUTED, lw=1.0,
               label=r"partícula libre, $6Dt$ con $D = k_\mathrm{B}T/\gamma M$")],
        loc="upper left", handlelength=1.6, borderaxespad=0.4)
    return _guardar(fig, salida)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", required=True, help="carpeta coords/ de la corrida")
    p.add_argument("--prefijo", required=True, help="p.ej. trp_b64")
    p.add_argument("--referencia", required=True, help="PDB experimental")
    p.add_argument("--psf", required=True, help="PSF que se simulo (de ahi salen las masas)")
    p.add_argument("--salida", required=True, help="carpeta de salida")
    p.add_argument("--molecula", default=None,
                   help="nombre para los titulos (por defecto, lo que va antes "
                        "del primer _ del prefijo)")
    p.add_argument("--dt", type=float, default=1.0, help="fs")
    p.add_argument("--temp", type=float, default=350.0, help="K")
    p.add_argument("--gamma", type=float, default=1.0, help="ps^-1")
    p.add_argument("--chunk", type=int, default=8, help="replicas por tanda en el RMSD")
    p.add_argument("--chunk-frames", type=int, default=20000,
                   help="frames por tanda en el RMSD y el COM; bajarlo si falta VRAM")
    p.add_argument("--rmsd-ymin", type=float, default=1.8)
    p.add_argument("--dist-xlim", nargs=2, type=float, default=None,
                   help="por defecto, percentiles 0.05 y 99.95 del dato")
    p.add_argument("--dist-ylim", nargs=2, type=float, default=None,
                   help="por defecto, el pico de la densidad con 12%% de aire")
    p.add_argument("--ancho-bin", type=float, default=0.1, help="A")
    p.add_argument("--alto", type=float, default=80.0,
                   help="mm, alto de la densidad y el MSD (el ancho es 170)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Agg solo en el camino de linea de comandos. Si se forzara al importar el
    # modulo, un notebook que haga `import fig_rmsd` perderia el backend inline
    # y las figuras dejarian de verse en la celda.
    matplotlib.use("Agg")

    from moleculekit.molecule import Molecule

    molecula = args.molecula or args.prefijo.split("_")[0]
    device = torch.device(args.device)
    salida = Path(args.salida)

    # El stride sale del propio archivo de pasos y no de una constante a mano:
    # un valor desfasado del real corre todo el eje temporal sin avisar.
    pasos = np.load(Path(args.coords) / f"{args.prefijo}_pasos.npy")
    stride = int(pasos[1] - pasos[0])
    t_ns = pasos * args.dt / 1e6

    ref = referencia_ca(args.referencia)
    rmsd_f = rmsd_motor(args.coords, args.prefijo, "fiumd", ref, device, args.chunk, args.chunk_frames)
    rmsd_t = rmsd_motor(args.coords, args.prefijo, "torchmd", ref, device, args.chunk, args.chunk_frames)

    masses = Molecule(args.psf).masses
    msd_f = msd_motor(args.coords, args.prefijo, "fiumd", masses, device, args.chunk_frames)
    msd_t = msd_motor(args.coords, args.prefijo, "torchmd", masses, device, args.chunk_frames)
    tau_ps = np.arange(msd_f.shape[1]) * stride * args.dt / 1000.0
    difusion = difusion_libre(masses, args.temp, args.gamma)

    generadas = [
        figura_rmsd(t_ns, rmsd_f, rmsd_t, molecula,
                    salida / f"figura_{args.prefijo}_rmsd", args.rmsd_ymin),
        figura_distribucion(rmsd_f, rmsd_t, molecula,
                            salida / f"figura_{args.prefijo}_distribucion",
                            args.ancho_bin,
                            tuple(args.dist_xlim) if args.dist_xlim else None,
                            tuple(args.dist_ylim) if args.dist_ylim else None,
                            args.alto * MM),
        figura_msd(tau_ps, msd_f, msd_t, difusion, molecula,
                   salida / f"figura_{args.prefijo}_msd", args.alto * MM),
    ]

    paso = max(1, rmsd_f.size // 50000)
    ks = ks_2samp(rmsd_f.ravel()[::paso], rmsd_t.ravel()[::paso]).statistic
    print(f"stride {stride} pasos = {stride * args.dt} fs | "
          f"M = {np.sum(masses):.0f} amu | D = {difusion:.4f} A^2/ps")
    print(f"RMSD   FiuMD {rmsd_f.mean():.2f} +- {rmsd_f.std():.2f} | "
          f"TorchMD {rmsd_t.mean():.2f} +- {rmsd_t.std():.2f} A | D_KS {ks:.4f}")
    for ruta in generadas:
        print(ruta)


if __name__ == "__main__":
    main()
