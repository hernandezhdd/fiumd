"""Figuras de una corrida de heterogeneous_ensemble.py, por especie.

Lee la carpeta <outdir>/<run_id> que deja heterogeneous_ensemble.py
(config.json + traj_<especie>.npz) y arma cinco figuras 2x2, un panel por
especie:

    temperatura   distribucion de T instantanea contra la Gamma teorica
    epot          trazas de energia potencial
    rmsd          trazas de RMSD C-alfa contra la estructura experimental
    rmsd_dist     densidad del RMSD
    msd           MSD del centro de masa / 6 D tau (una sola grafica)

y una tabla resumen (resumen.csv). `--torchmd` es opcional: con una sola
corrida las figuras muestran solo FiuMD; con las dos, cada panel las superpone
con el mismo codigo de color y trazo que fig_rmsd.py.

Uso:
    python benchmark/fig_sistemas_multiples.py \\
        --fiumd results/sistemas_multiples/het_mismo_dt_T_gamma \\
        --salida results/sistemas_multiples/het_mismo_dt_T_gamma/figuras

    # cuando este la corrida de TorchMD
    python benchmark/fig_sistemas_multiples.py \\
        --fiumd results/sistemas_multiples/het_mismo_dt_T_gamma \\
        --torchmd results/sistemas_multiples/<run_id_torchmd> \\
        --salida results/sistemas_multiples/het_mismo_dt_T_gamma/figuras
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gamma as gamma_dist, ks_2samp

from fig_rmsd import (
    COLOR_FIUMD, COLOR_TORCHMD, COLOR_MUTED, GUIONES, MM, ANCHO, _ejes,
    difusion_libre, estilo, referencia_ca,
)
from panel_proteinas import COLOR_PROTEINA, _grilla, _letra, _mostrar, msd_fft
from heterogeneous_ensemble import parse_temps

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mismas estructuras que usan fig_rmsd.py y panel_proteinas.py
REFERENCIAS = {
    "cln": "chignolin_cln025.pdb",
    "trp": "trpcage_2jof.pdb",
    "bbl": "bbl_2wxc.pdb",
    "lambda": "lambda_1lmb.pdb",
}

# Cada motor con su color y su trazo: el trazo es lo que los distingue cuando
# el color ya lo usa la proteina (figura de MSD).
ESTILO_MOTOR = {
    "fiumd": dict(color=COLOR_FIUMD, ls="-", nombre="FiuMD"),
    "torchmd": dict(color=COLOR_TORCHMD, ls=GUIONES, nombre="TorchMD"),
}


# ---------------------------------------------------------------------------
# Lectura y calculo
# ---------------------------------------------------------------------------
def leer_corrida(run_dir):
    """config.json y traj_<especie>.npz de una corrida, en el orden de la composicion."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    if not (run_dir / "COMPLETED").exists():
        print(f"aviso: {run_dir} no tiene COMPLETED, la corrida puede estar cortada")

    especies = list(dict.fromkeys(item.partition(":")[0].strip()
                                  for item in cfg["composition"].split(",")))
    # dt y gamma pueden venir por especie (--dts, --gammas)
    dts = parse_temps(cfg["dts"], especies, cfg["dt"], "--dts")
    gammas = parse_temps(cfg["gammas"], especies, cfg["gamma"], "--gammas")
    temps = parse_temps(cfg["temps"], especies, cfg["temp"])

    trajs = {}
    for key in especies:
        d = np.load(run_dir / f"traj_{key}.npz")
        trajs[key] = {k: d[k] for k in d.files}
        trajs[key].update(dt=dts[key], gamma=gammas[key], temp_obj=temps[key])
    return cfg, trajs


def rmsd(pos, ref, device):
    """(nframes, nrep, natoms, 3) -> (nrep, nframes). MDutils quiere replicas primero."""
    import MDutils
    traj = torch.from_numpy(np.ascontiguousarray(pos.transpose(1, 0, 2, 3)))
    return MDutils.metrics.RMSD(traj, ref, device).cpu().numpy()


def calcular(trajs, device, datos_dir):
    """Agrega a cada especie lo que necesitan las figuras: t, RMSD, MSD, D."""
    from moleculekit.molecule import Molecule

    for key, d in trajs.items():
        natoms = d["pos"].shape[2]
        ref = referencia_ca(REPO_ROOT / "datos" / "experimental_structures" / REFERENCIAS[key])
        if ref.shape[2] != natoms:
            raise ValueError(f"{key}: la referencia tiene {ref.shape[2]} CA y la trayectoria {natoms}")
        masses = Molecule(str(Path(datos_dir) / key / f"{key}.psf")).masses

        stride = int(d["steps"][1] - d["steps"][0])
        d["natoms"] = natoms
        d["t_ns"] = d["steps"] * d["dt"] / 1e6
        d["tau_ps"] = np.arange(len(d["steps"])) * stride * d["dt"] / 1000.0
        d["rmsd"] = rmsd(d["pos"], ref, device)
        com = np.einsum("frad,a->rfd", d["pos"].astype(np.float64), masses) / masses.sum()
        d["msd"] = msd_fft(com)
        d["difusion"] = difusion_libre(masses, d["temp_obj"], d["gamma"])
        d["masa"] = float(masses.sum())
    return trajs


def mascara_eventos(temp, temp_obj, natoms, cola=5):
    """(nframes, nrep) True en los frames de un evento y los `cola` siguientes.

    Evento: T por encima del cuantil 1 - 1e-6 de la Gamma, que por azar no
    aparece en 32000 frames. Los frames de despues se descartan porque el
    termostato tarda unos ps en sacar la energia que entro en el evento.
    """
    umbral = gamma_dist(a=1.5 * natoms, scale=2 * temp_obj / (3 * natoms)).ppf(1 - 1e-6)
    mascara = np.zeros(temp.shape, dtype=bool)
    for f, r in zip(*np.where(temp > umbral)):
        mascara[f:f + cola + 1, r] = True
    return mascara


def tabla_resumen(res):
    """Una fila por especie y motor. Con las dos corridas agrega el D_KS del RMSD."""
    filas = []
    for key in res["fiumd"]:
        for motor, trajs in res.items():
            d = trajs[key]
            k = len(d["tau_ps"]) // 10 - 1        # el lag mas largo que se grafica
            limpio = d["temp"][~mascara_eventos(d["temp"], d["temp_obj"], d["natoms"])]
            fila = {
                "especie": key,
                "motor": ESTILO_MOTOR[motor]["nombre"],
                "replicas": d["rmsd"].shape[0],
                "duracion (ns)": round(float(d["t_ns"][-1]), 2),
                "T objetivo (K)": d["temp_obj"],
                "<T> (K)": round(float(d["temp"].mean()), 2),
                "sd T (K)": round(float(d["temp"].std()), 2),
                "sd T sin eventos (K)": round(float(limpio.std()), 2),
                "sd T teorico (K)": round(float(d["temp_obj"] * np.sqrt(2 / (3 * d["natoms"]))), 2),
                # Frames sueltos con T muy por encima de la Gamma: no se ven en el
                # histograma pero inflan el sd. Con la Gamma de cln, P(T > 3 T_obj) ~ 1e-8.
                "frames T > 3 T_obj": int((d["temp"] > 3 * d["temp_obj"]).sum()),
                "T max (K)": round(float(d["temp"].max()), 0),
                "<Epot> (kcal/mol)": round(float(d["epot"].mean()), 2),
                "RMSD (A)": f"{d['rmsd'].mean():.2f} ± {d['rmsd'].std():.2f}",
                "MSD/6Dtau": round(float(d["msd"][:, k].mean() / (6 * d["difusion"] * d["tau_ps"][k])), 3),
            }
            if "torchmd" in res:
                fila["D_KS RMSD"] = round(float(ks_2samp(
                    res["fiumd"][key]["rmsd"].ravel(), res["torchmd"][key]["rmsd"].ravel()).statistic), 4)
            filas.append(fila)
    return filas


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _leyenda_motores(ax, motores, relleno=False, loc="upper right"):
    if len(motores) < 2:
        return
    handles = []
    for m in motores:
        e = ESTILO_MOTOR[m]
        if relleno and m == "fiumd":
            handles.append(Patch(facecolor=e["color"], alpha=0.22, edgecolor=e["color"],
                                 lw=0.9, label=e["nombre"]))
        else:
            handles.append(Line2D([], [], color=e["color"], ls=e["ls"], lw=1.2, label=e["nombre"]))
    ax.legend(handles=handles, loc=loc, ncol=len(handles), handlelength=1.4,
              columnspacing=1.0, borderaxespad=0.3)


def _titulo(ax, key, d, extra=""):
    ax.set_title(f"{key}  ({d['natoms']} CA, {d['rmsd'].shape[0]} réplicas){extra}", pad=3)


def _histogramas(ax, muestras, bins):
    """FiuMD relleno + contorno, TorchMD como linea punteada encima.

    Uno relleno y el otro como linea: dos rellenos se tapan, dos lineas del
    mismo trazo se confunden.
    """
    for motor, x in muestras.items():
        e = ESTILO_MOTOR[motor]
        if motor == "fiumd":
            ax.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.22,
                    lw=0, color=e["color"], zorder=2)
        ax.hist(x, bins=bins, density=True, histtype="step", lw=0.9, ls=e["ls"],
                color=e["color"], zorder=3 if motor == "fiumd" else 4)


def figura_temperatura(res, salida):
    """T instantanea contra su distribucion exacta bajo Langevin.

    Con 3N grados de libertad (los que usa temperature_per_system), la energia
    cinetica es Gamma(3N/2, kB T), asi que T_inst = 2K / (3N kB) es
    Gamma(3N/2, escala 2T/3N). La curva no tiene parametros libres: el ancho
    tiene que salir del numero de atomos. cln, con 10 CA, fluctua +-26 %.
    """
    motores = list(res)
    fig, axes = _grilla(len(res["fiumd"]))
    for ax, key, letra in zip(axes, res["fiumd"], "abcd"):
        d0 = res["fiumd"][key]
        n, T = d0["natoms"], d0["temp_obj"]
        teo = gamma_dist(a=1.5 * n, scale=2 * T / (3 * n))
        lo, hi = teo.ppf([5e-5, 1 - 5e-5])
        bins = np.linspace(lo, hi, 60)

        _histogramas(ax, {m: res[m][key]["temp"].ravel() for m in motores}, bins)
        x = np.linspace(lo, hi, 400)
        ax.plot(x, teo.pdf(x), color=COLOR_MUTED, lw=0.9, zorder=5)
        ax.axvline(T, color=COLOR_MUTED, lw=0.6, ls=":", zorder=1)

        _ejes(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, None)
        _titulo(ax, key, d0)
        _letra(ax, letra)
    for ax in axes[2:]:
        ax.set_xlabel("T instantánea (K)")
    for ax in axes[::2]:
        ax.set_ylabel(r"densidad (K$^{-1}$)")

    handles = [Patch(facecolor=COLOR_FIUMD, alpha=0.22, edgecolor=COLOR_FIUMD, lw=0.9, label="FiuMD")]
    if "torchmd" in res:
        handles.append(Line2D([], [], color=COLOR_TORCHMD, ls=GUIONES, lw=1.2, label="TorchMD"))
    handles.append(Line2D([], [], color=COLOR_MUTED, lw=1.0, label=r"Gamma($3N/2$), teórica"))
    axes[0].legend(handles=handles, loc="upper right", handlelength=1.4, borderaxespad=0.3)
    return _mostrar(fig, salida)


def _sd_bootstrap(temp, n=2000, seed=0):
    """Intervalo del 95 % del sd remuestreando replicas (las columnas), que son
    lo unico independiente: los frames de una replica estan correlacionados y,
    peor, un evento ocupa varios frames seguidos."""
    rng = np.random.default_rng(seed)
    nrep = temp.shape[1]
    sds = [np.nanstd(temp[:, rng.integers(0, nrep, nrep)]) for _ in range(n)]
    return np.percentile(sds, [2.5, 97.5])


def _sd_gamma_truncada(a, escala, q):
    """sd de Gamma(a, escala) condicionada a quedar bajo su cuantil q.

    Es lo que tiene que dar sacar los frames mas calientes de una muestra que
    sigue la Gamma: el sd baja aunque no haya ningun evento.
    """
    c = gamma_dist(a, scale=escala).ppf(q)
    m1 = a * escala * gamma_dist(a + 1, scale=escala).cdf(c) / q
    m2 = a * (a + 1) * escala ** 2 * gamma_dist(a + 2, scale=escala).cdf(c) / q
    return np.sqrt(m2 - m1 ** 2)


def figura_sigma_t(res, salida, ejemplo=None, kmax=200):
    """De donde sale el exceso de sigma_T: cuatro paneles.

    a  sigma_T / teorico, con todos los frames y sin los eventos. Las barras
       son bootstrap sobre replicas. La marca gris es lo que daria 3N-3.
    b  sigma_T al ir sacando los k frames mas calientes, dividido por lo que
       daria la Gamma truncada en el mismo cuantil. Si el exceso fuera un
       ensanchamiento de toda la distribucion (grados de libertad mal
       contados, termostato flojo) la curva quedaria arriba de 1 para todo k.
       Si son unos pocos frames, cae a 1 en cuanto se los saca.
    c  Cola de la distribucion: P(T > x) contra la Gamma. Los eventos aparecen
       como escalones muy a la derecha de donde la Gamma ya es 1e-6.
    d  Un evento en el tiempo: la serie de T de la replica con mas sd.
    """
    estilo()
    fig, axes = plt.subplots(2, 2, figsize=(ANCHO, 2 * 58 * MM), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    especies = list(res["fiumd"])
    ancho = 0.35 if len(res) > 1 else 0.0

    for i, key in enumerate(especies):
        color = COLOR_PROTEINA[i % len(COLOR_PROTEINA)]
        for j, m in enumerate(res):
            d = res[m][key]
            T, T0, n = d["temp"].astype(float), d["temp_obj"], d["natoms"]
            a, escala = 1.5 * n, 2 * T0 / (3 * n)
            sd_teo = T0 * np.sqrt(2 / (3 * n))
            mascara = mascara_eventos(T, T0, n)
            ls = ESTILO_MOTOR[m]["ls"]

            # a) total (hueco) y sin eventos (lleno)
            x = i + (j - (len(res) - 1) / 2) * ancho
            limpio = np.where(mascara, np.nan, T)
            for dx, datos, relleno in ((-0.1, T, "none"), (0.1, limpio, color)):
                sd = np.nanstd(datos)
                lo, hi = _sd_bootstrap(datos)
                ax_a.errorbar(x + dx, sd / sd_teo, yerr=[[(sd - lo) / sd_teo], [(hi - sd) / sd_teo]],
                              fmt="o", ms=4.5, mfc=relleno, mec=color, ecolor=color, elinewidth=0.8,
                              capsize=1.5, zorder=3)
            if j == 0:
                ax_a.plot([i - 0.3, i + 0.3], [np.sqrt(3 * n / (3 * n - 3))] * 2,
                          color=COLOR_MUTED, lw=0.9, zorder=2)

            # b) sacar los k frames mas calientes
            orden = np.sort(T.ravel())
            ks = np.unique(np.r_[0, np.geomspace(1, kmax, 60).astype(int)])
            obs = np.array([orden[:orden.size - k].std() for k in ks])
            teo = np.array([_sd_gamma_truncada(a, escala, 1 - k / orden.size) if k else sd_teo for k in ks])
            ax_b.plot(ks, obs / teo, color=color, lw=1.1, ls=ls)

            # c) supervivencia
            P = 1 - np.arange(orden.size) / orden.size
            ax_c.step(orden / T0, P, where="post", color=color, lw=1.0, ls=ls)
            if j == 0:
                xs = np.linspace(orden.min(), max(orden.max(), gamma_dist(a, scale=escala).ppf(1 - 1e-7)), 400)
                ax_c.plot(xs / T0, gamma_dist(a, scale=escala).sf(xs), color=color, lw=0.7,
                          ls=":", zorder=1)

    ax_a.axhline(1.0, color=COLOR_MUTED, lw=0.7, ls=GUIONES, zorder=1)
    ax_a.set_xticks(range(len(especies)), [f"{k}\n(N={res['fiumd'][k]['natoms']})" for k in especies])
    ax_a.set_xlim(-0.6, len(especies) - 0.4)
    ax_a.set_ylabel(r"$\sigma_T\ /\ \sigma_T^{\mathrm{teo}}$")
    ax_a.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="none", mec=COLOR_MUTED, label="todos los frames"),
        Line2D([], [], marker="o", ls="none", mfc=COLOR_MUTED, mec=COLOR_MUTED, label="sin eventos"),
        Line2D([], [], color=COLOR_MUTED, lw=0.9, label=r"si fuera $3N-3$")],
        loc="upper right", handlelength=1.2, borderaxespad=0.3)

    ax_b.axhline(1.0, color=COLOR_MUTED, lw=0.7, ls=GUIONES, zorder=1)
    ax_b.set_xscale("symlog", linthresh=1)
    ax_b.set_xlim(0, kmax)
    ax_b.set_xlabel(r"frames más calientes descartados, $k$")
    ax_b.set_ylabel(r"$\sigma_T\ /\ \sigma_T^{\mathrm{teo}}$ (Gamma truncada)")
    ax_b.legend(handles=[
        Line2D([], [], color=COLOR_PROTEINA[i % len(COLOR_PROTEINA)], lw=1.2, label=k)
        for i, k in enumerate(especies)], loc="upper right", handlelength=1.4, borderaxespad=0.3)

    ax_c.set_yscale("log")
    ax_c.set_ylim(0.8 / orden.size, 1.5)
    ax_c.set_xlim(1.0, None)
    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"$T\ /\ T_0$")
    ax_c.set_ylabel(r"$P(T_{\mathrm{inst}} > T)$")
    ax_c.legend(handles=[
        Line2D([], [], color=COLOR_MUTED, lw=1.0, label="simulación"),
        Line2D([], [], color=COLOR_MUTED, lw=0.7, ls=":", label="Gamma teórica")],
        loc="upper right", handlelength=1.4, borderaxespad=0.3)

    # d) la replica con mas sd de la especie con mas exceso
    if ejemplo is None:
        exceso = {k: res["fiumd"][k]["temp"].std() / (res["fiumd"][k]["temp_obj"]
                                                       * np.sqrt(2 / (3 * res["fiumd"][k]["natoms"])))
                  for k in especies}
        key = max(exceso, key=exceso.get)
        rep = int(res["fiumd"][key]["temp"].std(0).argmax())
    else:
        key, rep = ejemplo
    d = res["fiumd"][key]
    T, T0, n = d["temp"][:, rep], d["temp_obj"], d["natoms"]
    f = int(T.argmax())
    s = slice(max(f - 15, 0), f + 55)
    t_ps = d["steps"] * d["dt"] / 1000.0
    color = COLOR_PROTEINA[especies.index(key) % len(COLOR_PROTEINA)]
    teo = gamma_dist(1.5 * n, scale=2 * T0 / (3 * n))
    ax_d.fill_between(t_ps[s], *teo.ppf([0.005, 0.995]), color=COLOR_MUTED, alpha=0.15, lw=0,
                      label="99 % de la Gamma")
    ax_d.axhline(teo.ppf(1 - 1e-6), color=COLOR_MUTED, lw=0.7, ls=GUIONES,
                 label=r"cuantil $1-10^{-6}$ (umbral de evento)")
    ax_d.plot(t_ps[s], T[s], color=color, lw=1.0, marker="o", ms=2.2)
    ax_d.set_yscale("log")
    ax_d.set_yticks([300, 1000, 3000], ["300", "1000", "3000"])
    ax_d.set_yticks([], minor=True)
    ax_d.set_xlim(t_ps[s][0], t_ps[s][-1])
    ax_d.set_xlabel("tiempo (ps)")
    ax_d.set_ylabel(r"$T_{\mathrm{inst}}$ (K)")
    ax_d.legend(loc="upper right", handlelength=1.4, borderaxespad=0.3)
    ax_d.set_title(f"{key}, réplica {rep}", pad=3)

    for ax, letra in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        _ejes(ax, which="major")
        _letra(ax, letra)
    if len(res) > 1:
        _leyenda_motores(ax_b, list(res), loc="lower left")
    return _mostrar(fig, salida)


def _trazas(ax, res, key, campo):
    """Una linea fina por replica y la media del ensemble encima."""
    trazas = [(m, r) for m in res for r in _por_replica(res[m][key], campo)]
    # Orden mezclado: dibujar un motor entero y despues el otro deja al
    # segundo sistematicamente arriba, y parece mas denso.
    random.Random(0).shuffle(trazas)
    t = res["fiumd"][key]["t_ns"]
    for m, r in trazas:
        ax.plot(res[m][key]["t_ns"], r, lw=0.25, alpha=0.30,
                color=ESTILO_MOTOR[m]["color"], rasterized=True)
    for z, m in enumerate(res):
        e = ESTILO_MOTOR[m]
        ax.plot(res[m][key]["t_ns"], _por_replica(res[m][key], campo).mean(0),
                lw=1.1, color=e["color"], ls=e["ls"], zorder=3 + z)
    ax.set_xlim(t[0], t[-1])


def _por_replica(d, campo):
    """(nrep, nframes) para cualquier campo: rmsd ya viene asi, epot y temp no."""
    return d[campo] if campo == "rmsd" else d[campo].T


def figura_trazas(res, campo, ylabel, salida):
    fig, axes = _grilla(len(res["fiumd"]))
    for ax, key, letra in zip(axes, res["fiumd"], "abcd"):
        _trazas(ax, res, key, campo)
        _ejes(ax)
        _titulo(ax, key, res["fiumd"][key])
        _letra(ax, letra)
    for ax in axes[2:]:
        ax.set_xlabel("tiempo (ns)")
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)
    _leyenda_motores(axes[0], list(res))
    return _mostrar(fig, salida)


def figura_rmsd_dist(res, salida):
    """Densidad del RMSD, bins por regla de Scott sobre el desvio."""
    motores = list(res)
    fig, axes = _grilla(len(res["fiumd"]))
    for ax, key, letra in zip(axes, res["fiumd"], "abcd"):
        muestras = {m: res[m][key]["rmsd"].ravel() for m in motores}
        juntos = np.concatenate(list(muestras.values()))
        ancho = max(0.02, round(3.5 * juntos.std() / juntos.size ** (1 / 3), 2))
        lo, hi = np.percentile(juntos, [0.05, 99.95])
        xlim = (np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2)
        bins = np.arange(xlim[0], xlim[1] + ancho, ancho)

        _histogramas(ax, muestras, bins)
        _ejes(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, None)
        _titulo(ax, key, res["fiumd"][key], f", bin {ancho:g} " r"$\mathrm{\AA}$")
        _letra(ax, letra)
    for ax in axes[2:]:
        ax.set_xlabel(r"RMSD C$\alpha$ ($\mathrm{\AA}$)")
    for ax in axes[::2]:
        ax.set_ylabel(r"densidad ($\mathrm{\AA}^{-1}$)")
    _leyenda_motores(axes[0], motores, relleno=True)
    return _mostrar(fig, salida)


def figura_msd(res, salida, alto=80):
    """MSD del COM / 6 D tau, las cuatro especies juntas.

    Normalizado porque cada especie tiene su masa y su D: crudo serian rectas
    paralelas. Bajo 1 a tiempos cortos es el regimen balistico (tau < 1/gamma).
    Solo se grafica hasta 1/10 de la trayectoria: mas alla quedan pocos
    origenes y la curva es ruido.
    """
    estilo()
    fig, ax = plt.subplots(figsize=(ANCHO, alto * MM), layout="constrained")
    for i, key in enumerate(res["fiumd"]):
        color = COLOR_PROTEINA[i % len(COLOR_PROTEINA)]
        for m in res:
            d = res[m][key]
            s = slice(1, len(d["tau_ps"]) // 10)
            tau = d["tau_ps"][s]
            ax.plot(tau, d["msd"].mean(0)[s] / (6 * d["difusion"] * tau),
                    lw=1.1, color=color, ls=ESTILO_MOTOR[m]["ls"])

    ax.axhline(1.0, color=COLOR_MUTED, lw=0.7)
    ax.set_xscale("log")
    _ejes(ax, which="both")
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"MSD $/\ 6D\tau$")
    ax.set_ylim(0, 1.3)

    leyenda = ax.legend(handles=[
        Line2D([], [], color=COLOR_PROTEINA[i % len(COLOR_PROTEINA)], lw=1.2, label=key)
        for i, key in enumerate(res["fiumd"])],
        loc="lower right", ncol=len(res["fiumd"]), handlelength=1.4,
        columnspacing=1.0, borderaxespad=0.5)
    if "torchmd" in res:
        ax.add_artist(leyenda)
        ax.legend(handles=[
            Line2D([], [], color=COLOR_MUTED, lw=1.2, ls=ESTILO_MOTOR[m]["ls"],
                   label=ESTILO_MOTOR[m]["nombre"]) for m in res],
            loc="upper left", handlelength=1.6, borderaxespad=0.5)
    return _mostrar(fig, salida)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fiumd", required=True, help="carpeta de la corrida con --engine fiumd")
    p.add_argument("--torchmd", default=None, help="carpeta de la corrida con --engine torchmd (opcional)")
    p.add_argument("--salida", required=True, help="carpeta donde van los PNG y resumen.csv")
    p.add_argument("--datos_dir", default=str(REPO_ROOT / "datos"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    corridas = {"fiumd": args.fiumd}
    if args.torchmd:
        corridas["torchmd"] = args.torchmd

    res, cfgs = {}, {}
    for motor, run_dir in corridas.items():
        cfgs[motor], trajs = leer_corrida(run_dir)
        if cfgs[motor]["engine"] != motor:
            raise SystemExit(f"--{motor} apunta a una corrida con engine={cfgs[motor]['engine']}")
        res[motor] = calcular(trajs, args.device, args.datos_dir)

    if "torchmd" in res:
        distintos = [k for k in ("composition", "temp", "temps", "dt", "dts", "gamma", "gammas",
                                 "steps", "stride", "nnp", "precision")
                     if cfgs["fiumd"].get(k) != cfgs["torchmd"].get(k)]
        if distintos:
            print(f"aviso: las corridas difieren en {distintos}")

    salida = Path(args.salida)
    figura_temperatura(res, salida / "temperatura")
    figura_sigma_t(res, salida / "sigma_t")
    figura_trazas(res, "epot", r"$E_\mathrm{pot}$ (kcal/mol)", salida / "epot")
    figura_trazas(res, "rmsd", r"RMSD C$\alpha$ ($\mathrm{\AA}$)", salida / "rmsd")
    figura_rmsd_dist(res, salida / "rmsd_dist")
    figura_msd(res, salida / "msd")

    filas = tabla_resumen(res)
    with open(salida / "resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(salida / "resumen.csv")
    for fila in filas:
        print(fila)


if __name__ == "__main__":
    main()
