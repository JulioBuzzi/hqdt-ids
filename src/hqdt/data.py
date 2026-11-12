"""Download e pré-processamento do NSL-KDD, seguindo o artigo:
Label Encoding nas categóricas -> StandardScaler -> PCA (3 componentes = 3 qubits).
Rótulo binário: 0 = normal, 1 = ataque.
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download(data_dir=DATA_DIR):
    """Baixa KDDTrain+ e KDDTest+ se ainda não estiverem em data/."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    for fname in FILES.values():
        path = data_dir / fname
        if not path.exists():
            print(f"baixando {fname}...")
            urllib.request.urlretrieve(BASE_URL + fname, path)
    return data_dir


def load(split="train", data_dir=DATA_DIR):
    """Lê o arquivo bruto e cria a coluna `label` (0 = normal, 1 = ataque)."""
    download(data_dir)
    df = pd.read_csv(Path(data_dir) / FILES[split], header=None, names=COLUMNS)
    df["label"] = (df["attack"] != "normal").astype(int)
    return df


def stratified_sample(df, n, seed=0):
    """Amostra estratificada de n linhas, mantendo a proporção normal/ataque."""
    if n >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample


class Preprocessor:
    """Label encoding + StandardScaler + PCA. Ajustado só no treino (sem vazamento)."""

    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, df):
        self.features_ = [c for c in COLUMNS if c not in ("attack", "difficulty")]
        # OrdinalEncoder = LabelEncoder pra várias colunas; categoria nova no teste vira -1
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder_.fit(df[CATEGORICAL])
        X = self._encode(df)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(self.scaler_.transform(X))
        return self

    def _encode(self, df):
        X = df[self.features_].copy()
        X[CATEGORICAL] = self.encoder_.transform(df[CATEGORICAL])
        return X.to_numpy(dtype=float)

    def transform(self, df):
        return self.pca_.transform(self.scaler_.transform(self._encode(df)))

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def make_split(n, seed=0, test_size=0.3, n_components=3):
    """Protocolo dos experimentos: amostra n do KDDTrain+, divide 70/30 estratificado,
    ajusta o pré-processamento no treino. Retorna X_train, X_test, y_train, y_test."""
    df = stratified_sample(load("train"), n, seed=seed)
    tr, te = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    prep = Preprocessor(n_components).fit(tr)
    return prep.transform(tr), prep.transform(te), tr["label"].to_numpy(), te["label"].to_numpy()