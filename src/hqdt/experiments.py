"""Experimentos do notebook 03: compara o HQDT com árvores clássicas e ablações."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from .data import Preprocessor, load, make_split, stratified_sample
from .tree import HybridQuantumDecisionTree


def get_models(max_depth=4, seed=0):
    """Modelos comparados. As duas primeiras são as baselines do artigo;
    as duas últimas são ablações pra isolar o efeito da parte quântica."""
    return {
        "Árvore clássica (sem limite)": DecisionTreeClassifier(random_state=seed),
        f"Árvore clássica (depth={max_depth})": DecisionTreeClassifier(max_depth=max_depth, random_state=seed),
        f"HQDT quântica (depth={max_depth}, alpha=0.5)": HybridQuantumDecisionTree(max_depth, alpha=0.5, distance="quantum"),
        f"Ablação: só IG, percentis (depth={max_depth}, alpha=1)": HybridQuantumDecisionTree(max_depth, alpha=1.0, distance="none"),
        f"Ablação: distância RBF clássica (depth={max_depth}, alpha=0.5)": HybridQuantumDecisionTree(max_depth, alpha=0.5, distance="rbf"),
    }


def _metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


def _depth(model):
    return model.get_depth()


def run_sizes(sizes=(1000, 5000, 10000, 25000), seeds=(0, 1, 2, 3, 4), max_depth=4, verbose=True):
    """Para cada tamanho e semente: amostra estratificada do KDDTrain+, split 70/30,
    treina todos os modelos e mede no conjunto de teste."""
    rows = []
    for n in sizes:
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = make_split(n, seed=seed)
            for name, model in get_models(max_depth, seed).items():
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                fit_time = time.perf_counter() - t0
                rows.append({"n": n, "seed": seed, "modelo": name, "fit_time_s": fit_time,
                             "depth": _depth(model), "folhas": model.get_n_leaves(),
                             **_metrics(model, X_te, y_te)})
            if verbose:
                print(f"n={n} seed={seed} ok")
    return pd.DataFrame(rows)


def run_official_test(n_train=25000, seeds=(0, 1, 2), max_depth=4):
    """Treina em uma amostra do KDDTrain+ e testa no KDDTest+ (o teste oficial,
    que tem tipos de ataque que não aparecem no treino)."""
    test_df = load("test")
    rows = []
    for seed in seeds:
        train_df = stratified_sample(load("train"), n_train, seed=seed)
        prep = Preprocessor(3).fit(train_df)
        X_tr, y_tr = prep.transform(train_df), train_df["label"].to_numpy()
        X_te, y_te = prep.transform(test_df), test_df["label"].to_numpy()
        for name, model in get_models(max_depth, seed).items():
            model.fit(X_tr, y_tr)
            rows.append({"seed": seed, "modelo": name, **_metrics(model, X_te, y_te)})
    return pd.DataFrame(rows)


def summarize(df, metric="accuracy"):
    """Tabela média ± desvio padrão (em %) por modelo e tamanho."""
    g = df.groupby(["modelo", "n"])[metric]
    table = (g.mean() * 100).round(1).astype(str) + " ± " + (g.std() * 100).round(1).astype(str)
    return table.unstack("n")