"""
SMOTE - Synthetic Minority Over-sampling Technique
==================================================

Ideja në një fjali:
    Për çdo mostër të klasës së pakicës, zgjidh një nga fqinjët e saj
    më të afërt (po të pakicës) dhe krijo një pikë të re diku në vijën
    që i bashkon të dyja.

Formula:
    x_ri = x + λ · (x_fqinj − x),    ku λ ~ Uniform(0, 1)

Hapat:
    1. Merr vetëm mostrat e klasës së pakicës.
    2. Për secilën, gjej k fqinjët më të afërt (distanca Euklidiane).
    3. Përsërit derisa të krijohen aq mostra sa duhen:
         a. zgjidh një mostër x rastësisht
         b. zgjidh një nga k fqinjët e saj rastësisht
         c. krijo pikën e re me formulën më sipër
"""

import numpy as np


def k_fqinjet_me_te_afert(X, k):
    """Kthen indekset e k fqinjëve më të afërt për çdo rresht të X."""
    # Matrica e distancave: D[i, j] = distanca mes X[i] dhe X[j]
    # Përdorim identitetin  ||a - b||² = ||a||² + ||b||² - 2·a·b
    # (i njëjti rezultat, por pa krijuar një matricë gjigante n × n × veçori)
    katroret = (X ** 2).sum(axis=1)
    D2 = katroret[:, None] + katroret[None, :] - 2 * X @ X.T
    D = np.sqrt(np.maximum(D2, 0))          # maximum: shmang -0.0000001 nga rrumbullakimi

    # Një pikë nuk është fqinja e vetvetes
    np.fill_diagonal(D, np.inf)

    # Renditim distancat dhe mbajmë k më të voglat
    return np.argsort(D, axis=1)[:, :k]


def smote(X_pakica, n_te_reja, k=5, random_state=None):
    """
    Gjeneron `n_te_reja` mostra sintetike nga mostrat e pakicës.

    Parametrat
    ----------
    X_pakica    : array (n_mostra, n_veçori) - vetëm mostrat e pakicës
    n_te_reja   : sa mostra sintetike të krijohen
    k           : numri i fqinjëve që merren parasysh
    random_state: për rezultate të përsëritshme

    Kthen
    -----
    array (n_te_reja, n_veçori) me mostrat sintetike
    """
    rng = np.random.default_rng(random_state)
    k = min(k, len(X_pakica) - 1)          # s'mund të kemi më shumë fqinjë se pika

    fqinjet = k_fqinjet_me_te_afert(X_pakica, k)

    # (a) zgjedhim rastësisht mostrat bazë
    baza = rng.integers(0, len(X_pakica), size=n_te_reja)
    # (b) për secilën zgjedhim rastësisht një nga k fqinjët
    fqinji = fqinjet[baza, rng.integers(0, k, size=n_te_reja)]
    # (c) λ ∈ [0, 1) - sa larg drejt fqinjit lëvizim
    lam = rng.random((n_te_reja, 1))

    return X_pakica[baza] + lam * (X_pakica[fqinji] - X_pakica[baza])


def balanco_me_smote(X, y, k=5, random_state=None):
    """
    Balancon një dataset binar: krijon aq mostra sintetike për klasën
    e pakicës sa që të barazohet me klasën e shumicës.
    """
    klasat, numrat = np.unique(y, return_counts=True)
    klasa_pakices = klasat[np.argmin(numrat)]
    n_te_reja = numrat.max() - numrat.min()

    X_pakica = X[y == klasa_pakices]
    X_sintetik = smote(X_pakica, n_te_reja, k=k, random_state=random_state)
    y_sintetik = np.full(n_te_reja, klasa_pakices)

    return np.vstack([X, X_sintetik]), np.concatenate([y, y_sintetik])
