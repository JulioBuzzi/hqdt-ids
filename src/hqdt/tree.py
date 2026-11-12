"""Árvore de Decisão Quântica Híbrida (HQDT).

Critério de divisão: S(f, t) = alpha * IG + (1 - alpha) * d
  - IG: ganho de informação de Shannon do corte t na feature f
  - d : distância entre a média da partição da esquerda e o centróide do nó.
        No HQDT, d = 1 - fidelidade quântica (ZZFeatureMap + FidelityQuantumKernel).

O parâmetro `distance` deixa trocar a distância quântica por uma clássica (RBF)
ou desligar a distância (alpha=1 vira uma árvore só com IG). Isso serve pros
experimentos de ablação no notebook 03.
"""

import time
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Node:
    """Um nó da árvore. Se `value` não é None, é uma folha."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class HybridQuantumDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Parâmetros
    ----------
    max_depth : profundidade máxima da árvore.
    alpha : peso do ganho de informação (1 - alpha vai pra distância).
    distance : "quantum" (fidelidade), "rbf" (kernel clássico) ou "none".
    reps : repetições do ZZFeatureMap.
    percentiles : percentis usados como limiares candidatos em cada feature.
    rbf_gamma : gamma do kernel RBF (só usado com distance="rbf").
    use_cache : guarda fidelidades já calculadas.
    """

    def __init__(self, max_depth=4, alpha=0.5, distance="quantum", reps=1,
                 percentiles=(20, 40, 60, 80), rbf_gamma=1.0, use_cache=True):
        self.max_depth = max_depth
        self.alpha = alpha
        self.distance = distance
        self.reps = reps
        self.percentiles = percentiles
        self.rbf_gamma = rbf_gamma
        self.use_cache = use_cache

    # ------------------------------------------------------------------ treino
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)  # bincount precisa de rótulos inteiros >= 0

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.kernel_cache_ = {}
        self.n_kernel_calls_ = 0   # quantas vezes o kernel foi realmente calculado
        self.n_cache_hits_ = 0

        if self.distance == "quantum":
            # imports aqui pra quem só usa a versão clássica não precisar do qiskit
            from qiskit.circuit.library import zz_feature_map
            from qiskit_machine_learning.kernels import FidelityQuantumKernel

            feature_map = zz_feature_map(feature_dimension=self.n_features_in_, reps=self.reps)
            self.quantum_kernel_ = FidelityQuantumKernel(feature_map=feature_map)

        start = time.perf_counter()
        self.root_ = self._grow_tree(X, y, depth=0)
        self.fit_time_ = time.perf_counter() - start
        return self

    def _similarity(self, x1, x2):
        """Similaridade entre dois vetores, entre 0 e 1."""
        key = tuple(x1) + tuple(x2)
        key_rev = tuple(x2) + tuple(x1)  # F(a, b) = F(b, a)
        if self.use_cache:
            if key in self.kernel_cache_:
                self.n_cache_hits_ += 1
                return self.kernel_cache_[key]
            if key_rev in self.kernel_cache_:
                self.n_cache_hits_ += 1
                return self.kernel_cache_[key_rev]

        self.n_kernel_calls_ += 1
        if self.distance == "quantum":
            value = self.quantum_kernel_.evaluate(x1.reshape(1, -1), x2.reshape(1, -1))[0][0]
        else:  # rbf
            value = np.exp(-self.rbf_gamma * np.sum((x1 - x2) ** 2))

        if self.use_cache:
            self.kernel_cache_[key] = value
        return value

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=self._most_common_label(y))

        best_feature, best_threshold = self._best_split(X, y)

        # nenhum corte com ganho > 0 -> vira folha
        # (no código original isso quebrava com X[:, None])
        if best_feature is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_score = -1
        split_idx, split_threshold = None, None
        centroid = np.mean(X, axis=0)  # centróide do nó (não muda dentro do loop)

        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            thresholds = np.unique(np.percentile(X_column, self.percentiles))

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain == 0:
                    continue

                if self.distance == "none" or self.alpha == 1:
                    dist = 0.0
                else:
                    left_idxs, _ = self._split(X_column, threshold)
                    x_left = np.mean(X[left_idxs], axis=0)
                    dist = 1 - self._similarity(x_left, centroid)

                score = self.alpha * gain + (1 - self.alpha) * dist
                if score > best_score:
                    best_score = score
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    @staticmethod
    def _split(X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    @staticmethod
    def _entropy(y):
        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain(self, y, X_column, threshold):
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        child = (n_l / n) * self._entropy(y[left_idxs]) + (n_r / n) * self._entropy(y[right_idxs])
        return self._entropy(y) - child

    @staticmethod
    def _most_common_label(y):
        return Counter(y).most_common(1)[0][0]

    # -------------------------------------------------------------- previsão
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(x, self.root_) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    # ------------------------------------------------------------ utilidades
    def get_depth(self):
        def depth(node):
            return 0 if node.is_leaf() else 1 + max(depth(node.left), depth(node.right))
        return depth(self.root_)

    def get_n_leaves(self):
        def leaves(node):
            return 1 if node.is_leaf() else leaves(node.left) + leaves(node.right)
        return leaves(self.root_)

    def print_tree(self, feature_names=None, class_names=None):
        """Mostra a árvore em texto, parecido com sklearn.tree.export_text."""
        lines = []

        def name(i):
            return feature_names[i] if feature_names is not None else f"x{i}"

        def label(v):
            return class_names[v] if class_names is not None else str(v)

        def walk(node, indent):
            if node.is_leaf():
                lines.append(f"{indent}-> {label(node.value)}")
                return
            lines.append(f"{indent}{name(node.feature)} <= {node.threshold:.3f}")
            walk(node.left, indent + "|   ")
            lines.append(f"{indent}{name(node.feature)} >  {node.threshold:.3f}")
            walk(node.right, indent + "|   ")

        walk(self.root_, "")
        return "\n".join(lines)