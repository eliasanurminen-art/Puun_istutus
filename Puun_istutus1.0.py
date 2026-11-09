"""
Puun_istutus 1.0 (PI)
=====================

Luo satunnaisen, pienen kaukolämpöverkon kokoisen puumallin ja lisää siihen
yhden ylimääräisen linkin (SML) kahden lehtisolmun välille, jolloin verkkoon
syntyy tasan yksi sykli. (Kandityö, luku 3.1.)

Putkien halkaisijat määritetään verkon rakenteen perusteella (luku 3.2):
  * kuluttajasolmuun saapuva putki on DN32
  * ylävirtaan kohti 0-solmua:  D_sisään = sqrt( sum(D_ulos^2) / n_sisään )

Tulokset tallennetaan tiedostoon network_data.npz:
  edges          (l, 2)  linkit (u, v) nimellissuunnassa u -> v
  edge_features  (l, 6)  [L, D, A, karheus, u, v]
  adjacency      (l, m)  yhdistymismatriisi: alkusolmu -1, loppusolmu +1
  kuluttajat     (k,)    kuluttajasolmujen indeksit (puumallin lehdet)
  sml_indeksit   (s,)    ylimääräisten linkkien indeksit edges-taulukossa

Verkon nimellissuunta: puu kasvatetaan 0-solmusta lähtien niin, että uusi
solmu saa aina suuremman indeksin kuin vanhempansa. Kaikki linkit kulkevat
siis pienemmästä indeksistä suurempaan (u < v). Myös SML suunnataan näin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# PARAMETRIT (muuta näitä tarvittaessa)
# ----------------------------------------------------------------------------
LINKKEJA_MIN = 20           # puumallin linkkien määrä l arvotaan väliltä
LINKKEJA_MAX = 100
ASTE_MIN = 1                # jokainen solmu yhteydessä 1...4 muuhun solmuun
ASTE_MAX = 4
PITUUS_MIN, PITUUS_MAX = 100.0, 5000.0      # m
KARHEUS_MIN, KARHEUS_MAX = 1e-5, 1e-4       # m (luku 2.2)
D_DN32 = 0.032              # m, kuluttajaan saapuvan putken halkaisija (DN32)
SML_MAARA = 1               # kuinka monta ylimääräistä linkkiä lisätään
TIEDOSTO = "network_data.npz"


@dataclass
class Verkko:
    n_solmut: int
    edges: np.ndarray            # (l, 2)
    edge_features: np.ndarray    # (l, 6)
    adjacency: np.ndarray        # (l, m)
    kuluttajat: np.ndarray       # lehtisolmut
    sml_indeksit: np.ndarray     # SML:ien indeksit


# ============================================================================
# 1) PUUMALLIN MUODOSTAMINEN
# ============================================================================
def _yrita_puuta(n_solmut: int, rng: random.Random) -> Optional[List[Tuple[int, int]]]:
    """Kasvattaa puun 0-solmusta. Uusi solmu liitetään satunnaiseen jo puussa
    olevaan solmuun, jonka aste on vielä alle ASTE_MAX. Palauttaa None, jos
    asteraja estää puun muodostamisen."""
    aste = [0] * n_solmut
    linkit = []
    puussa = [0]

    for uusi in range(1, n_solmut):
        ehdokkaat = [s for s in puussa if aste[s] < ASTE_MAX]
        if not ehdokkaat:
            return None
        vanhempi = rng.choice(ehdokkaat)
        linkit.append((vanhempi, uusi))          # vanhempi < uusi aina
        aste[vanhempi] += 1
        aste[uusi] += 1
        puussa.append(uusi)

    if any(a < ASTE_MIN or a > ASTE_MAX for a in aste):
        return None
    return sorted(linkit)


def luo_puumalli(rng: random.Random, max_yritykset: int = 5000) -> Tuple[int, List[Tuple[int, int]]]:
    """Arpoo linkkien määrän l ja luo puun, jossa on m = l + 1 solmua."""
    l = rng.randint(LINKKEJA_MIN, LINKKEJA_MAX)
    n_solmut = l + 1
    for _ in range(max_yritykset):
        linkit = _yrita_puuta(n_solmut, rng)
        if linkit is not None:
            return n_solmut, linkit
    raise RuntimeError("Puun luominen epäonnistui asterajoilla.")


def etsi_lehdet(n_solmut: int, linkit: List[Tuple[int, int]]) -> np.ndarray:
    """Lehtisolmu = solmu, johon nimellissuunnassa vain saapuu linkkejä."""
    lahtee = np.zeros(n_solmut, dtype=bool)
    for u, _ in linkit:
        lahtee[u] = True
    lehdet = np.where(~lahtee)[0]
    return lehdet[lehdet != 0]


# ============================================================================
# 2) YLIMÄÄRÄISEN LINKIN (SML) LISÄÄMINEN
# ============================================================================
def lisaa_sml(linkit: List[Tuple[int, int]], lehdet: np.ndarray,
              rng: random.Random, maara: int = SML_MAARA) -> List[Tuple[int, int]]:
    """Lisää 'maara' kpl linkkejä kahden satunnaisen lehtisolmun välille.
    Linkki suunnataan pienemmästä indeksistä suurempaan."""
    olemassa = set(linkit)
    uudet = []
    vapaat = list(lehdet)
    if len(vapaat) < 2 * maara:
        raise RuntimeError("Lehtisolmuja on liian vähän SML:ien lisäämiseen.")

    while len(uudet) < maara:
        a, b = rng.sample(vapaat, 2)
        linkki = (min(a, b), max(a, b))
        if linkki in olemassa:
            continue
        olemassa.add(linkki)
        uudet.append(linkki)
        vapaat.remove(a)            # sama lehti ei saa kahta SML:ää
        vapaat.remove(b)
    return uudet


# ============================================================================
# 3) PUTKIEN HALKAISIJAT (luku 3.2)
# ============================================================================
def topologinen_jarjestys(n_solmut: int, edges: np.ndarray) -> np.ndarray:
    """Kahnin algoritmi: solmut järjestyksessä, jossa jokainen linkki u -> v
    tulee ennen v:tä. Käänteinen järjestys = lehdistä kohti 0-solmua."""
    sisaan = np.zeros(n_solmut, dtype=int)
    ulos = [[] for _ in range(n_solmut)]
    for u, v in edges:
        sisaan[v] += 1
        ulos[u].append(v)
    jono = [s for s in range(n_solmut) if sisaan[s] == 0]
    jarjestys = []
    while jono:
        s = jono.pop(0)
        jarjestys.append(s)
        for v in ulos[s]:
            sisaan[v] -= 1
            if sisaan[v] == 0:
                jono.append(v)
    if len(jarjestys) != n_solmut:
        raise RuntimeError("Verkossa on suunnattu kehä, topologista järjestystä ei ole.")
    return np.array(jarjestys)


def maarita_halkaisijat(n_solmut: int, edges: np.ndarray, kuluttajat: np.ndarray) -> np.ndarray:
    """
    Käydään solmut läpi lehdistä kohti 0-solmua (käänteinen topologinen järjestys).
    Jokaisessa solmussa:
        poistuva_pinta = sum(D_ulos^2)  (+ D_DN32^2, jos solmu on kuluttaja)
        D_sisään       = sqrt(poistuva_pinta / n_sisään)
    Kuluttaja käsitellään siis "virtuaalisena" DN32-lähtöputkena. Tavallisessa
    lehdessä (1 saapuva putki) tämä antaa suoraan DN32, ja SML:n päissä pinta-ala
    jakautuu samalla tavalla kuin alkuarvaus jakaa tilavuusvirran.
    """
    l = len(edges)
    D = np.full(l, np.nan)
    kuluttaja = np.zeros(n_solmut, dtype=bool)
    kuluttaja[kuluttajat] = True

    sisaan_linkit = [[] for _ in range(n_solmut)]
    ulos_linkit = [[] for _ in range(n_solmut)]
    for i, (u, v) in enumerate(edges):
        ulos_linkit[u].append(i)
        sisaan_linkit[v].append(i)

    for solmu in topologinen_jarjestys(n_solmut, edges)[::-1]:
        if not sisaan_linkit[solmu]:
            continue                        # 0-solmu
        D2_ulos = sum(D[i] ** 2 for i in ulos_linkit[solmu])
        if kuluttaja[solmu]:
            D2_ulos += D_DN32 ** 2
        D_sisaan = np.sqrt(D2_ulos / len(sisaan_linkit[solmu]))
        for i in sisaan_linkit[solmu]:
            D[i] = D_sisaan

    if np.isnan(D).any():
        raise RuntimeError("Kaikille putkille ei saatu halkaisijaa.")
    return D


# ============================================================================
# 4) KOKO VERKON LUONTI JA TALLENNUS
# ============================================================================
def yhdistymismatriisi(edges: np.ndarray, n_solmut: int) -> np.ndarray:
    """(l x m) matriisi: rivi = linkki, alkusolmu -1, loppusolmu +1."""
    A12 = np.zeros((len(edges), n_solmut), dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        A12[i, u] = -1
        A12[i, v] = 1
    return A12


def luo_verkko(seed: Optional[int] = 42, tallenna: bool = True) -> Verkko:
    rng = random.Random(seed)

    # --- Puumalli ---
    n_solmut, puun_linkit = luo_puumalli(rng)
    kuluttajat = etsi_lehdet(n_solmut, puun_linkit)
    print(f"[PI] Puumalli: {n_solmut} solmua, {len(puun_linkit)} linkkiä, "
          f"{len(kuluttajat)} lehtisolmua (kuluttajaa)")

    # --- SML ---
    sml = lisaa_sml(puun_linkit, kuluttajat, rng)
    kaikki = puun_linkit + sml
    edges = np.array(kaikki, dtype=np.int32)
    sml_indeksit = np.arange(len(puun_linkit), len(kaikki), dtype=np.int32)
    for i in sml_indeksit:
        print(f"[PI] Lisätty SML: linkki {i} = {tuple(int(x) for x in edges[i])} (kahden lehden välillä)")

    # --- Attribuutit ---
    l = len(edges)
    L = np.array([rng.uniform(PITUUS_MIN, PITUUS_MAX) for _ in range(l)])
    eps = np.array([rng.uniform(KARHEUS_MIN, KARHEUS_MAX) for _ in range(l)])
    D = maarita_halkaisijat(n_solmut, edges, kuluttajat)
    A = np.pi * D ** 2 / 4.0

    edge_features = np.column_stack((L, D, A, eps, edges[:, 0], edges[:, 1])).astype(np.float64)
    adjacency = yhdistymismatriisi(edges, n_solmut)

    print(f"[PI] Halkaisijat: min {D.min()*1000:.1f} mm, max {D.max()*1000:.1f} mm "
          f"(0-solmusta lähtevät: {', '.join(f'{d*1000:.1f}' for d in D[edges[:, 0] == 0])} mm)")
    print(f"[PI] Pituudet: {L.min():.0f}–{L.max():.0f} m, karheudet: {eps.min():.2e}–{eps.max():.2e} m")

    if tallenna:
        np.savez(TIEDOSTO, edges=edges, edge_features=edge_features, adjacency=adjacency,
                 kuluttajat=kuluttajat, sml_indeksit=sml_indeksit)
        print(f"[PI] Tallennettu {TIEDOSTO}: edges {edges.shape}, "
              f"edge_features {edge_features.shape}, adjacency {adjacency.shape}")

    return Verkko(n_solmut, edges, edge_features, adjacency, kuluttajat, sml_indeksit)


# ============================================================================
# 5) PIIRTO
# ============================================================================
def piirra_verkko(verkko: Verkko, juuri: int = 0):
    """0-solmu punaisella, kuluttajat oranssilla, SML vihreällä.
    Viivan paksuus kuvaa putken halkaisijaa."""
    G = nx.DiGraph()
    G.add_nodes_from(range(verkko.n_solmut))
    G.add_edges_from(map(tuple, verkko.edges))
    pos = nx.kamada_kawai_layout(G.to_undirected())

    sml = set(verkko.sml_indeksit.tolist())
    D = verkko.edge_features[:, 1]
    leveys = 1 + 6 * (D - D.min()) / max(D.max() - D.min(), 1e-12)

    puu_e = [tuple(e) for i, e in enumerate(verkko.edges) if i not in sml]
    puu_w = [leveys[i] for i in range(len(verkko.edges)) if i not in sml]
    sml_e = [tuple(verkko.edges[i]) for i in sml]

    kuluttaja = set(verkko.kuluttajat.tolist())
    varit = ["red" if s == juuri else "orange" if s in kuluttaja else "lightblue" for s in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, edgelist=puu_e, width=puu_w, edge_color="gray", arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=sml_e, width=3, edge_color="green", arrows=True)
    nx.draw_networkx_nodes(G, pos, node_color=varit, node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"Verkko: {verkko.n_solmut} solmua, {len(verkko.edges)} linkkiä, "
              f"{len(sml)} SML (vihreä)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    verkko = luo_verkko(seed=42)

    print("\n[PI] Ensimmäinen linkki:", tuple(int(x) for x in verkko.edges[0]))
    print("     [L, D, A, karheus, u, v] =", np.array2string(verkko.edge_features[0], precision=5))
    piirra_verkko(verkko)


if __name__ == "__main__":
    main()
