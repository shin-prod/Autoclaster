#!/usr/bin/env python3
"""
自動クラスタリング探索ツール

CSVまたはExcelファイルを読み込み、カラムの型判定・前処理・クラスタリング手法と
パラメータの組み合わせを全自動で探索し、Silhouette係数をもとに結果を比較・可視化する。
"""

import argparse
import sys
import warnings
from pathlib import Path

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------- 再現性 ----------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------- matplotlib 日本語対応 ----------
plt.rcParams["font.family"] = [
    "IPAexGothic", "Noto Sans CJK JP", "Hiragino Sans",
    "Yu Gothic", "Meiryo", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. ファイル読み込み
# ============================================================
def load_data(filepath: str) -> pd.DataFrame:
    p = Path(filepath)
    if p.suffix == ".csv":
        return pd.read_csv(filepath)
    elif p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        sys.exit(f"[エラー] 未対応のファイル形式: {p.suffix}")


# ============================================================
# 2. 前処理パイプライン
# ============================================================

# 2-1. カラム型の自動判定
def detect_column_types(df: pd.DataFrame) -> dict:
    """各カラムの型を判定して dict で返す。"""
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            col_types[col] = "unknown"
            continue
        # ① 日付判定
        if _is_date_column(series):
            col_types[col] = "date"
            continue
        # ② 数値判定
        if _is_numeric_column(series):
            col_types[col] = "numeric"
            continue
        # ③ カテゴリ
        col_types[col] = "category"
    return col_types


def _is_date_column(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_string_dtype(series):
        sample = series.head(min(50, len(series)))
        try:
            parsed = pd.to_datetime(sample, format="mixed")
            return parsed.notna().mean() > 0.8
        except (ValueError, TypeError):
            return False
    return False


def _is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_string_dtype(series):
        try:
            pd.to_numeric(series)
            return True
        except (ValueError, TypeError):
            return False
    return False


# 2-2. 自動除外ルール
def columns_to_exclude(df: pd.DataFrame, col_types: dict) -> list:
    excluded = []
    for col in df.columns:
        series = df[col]
        n = len(series)
        n_unique = series.nunique(dropna=True)
        n_missing = series.isna().sum()

        # ユニーク率 > 80%
        if n > 0 and n_unique / n > 0.8:
            excluded.append(col)
            continue
        # 欠損率 > 50%
        if n > 0 and n_missing / n > 0.5:
            excluded.append(col)
            continue
        # 分散 = 0（数値のみ）
        if col_types.get(col) == "numeric":
            num = pd.to_numeric(df[col], errors="coerce")
            if num.dropna().std() == 0:
                excluded.append(col)
                continue
    return excluded


# 2-3. 型別エンコーディング
def encode_columns(df: pd.DataFrame, col_types: dict, excluded: list) -> pd.DataFrame:
    """前処理済み特徴量 DataFrame を返す。"""
    frames = []
    used_cols = [c for c in df.columns if c not in excluded]

    for col in used_cols:
        ctype = col_types.get(col, "unknown")
        if ctype == "numeric":
            frames.append(_encode_numeric(df, col))
        elif ctype == "date":
            frames.append(_encode_date(df, col))
        elif ctype == "category":
            result = _encode_category(df, col)
            if result is not None:
                frames.append(result)

    if not frames:
        sys.exit("[エラー] 有効な特徴量がありません。")

    encoded = pd.concat(frames, axis=1)
    # 残存 NaN を 0 埋め
    encoded = encoded.fillna(0)
    return encoded


def _encode_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    values = pd.to_numeric(df[col], errors="coerce").values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.nan_to_num(values, nan=0.0))
    return pd.DataFrame(scaled, columns=[col], index=df.index)


def _encode_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce")
    min_date = dt.min()
    features = pd.DataFrame(index=df.index)
    features[f"{col}_elapsed_days"] = (dt - min_date).dt.days.fillna(0)
    features[f"{col}_year"] = dt.dt.year.fillna(0)
    features[f"{col}_month"] = dt.dt.month.fillna(0)
    features[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0)
    features[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)
    return pd.DataFrame(scaled, columns=features.columns, index=df.index)


def _encode_category(df: pd.DataFrame, col: str) -> pd.DataFrame | None:
    n_unique = df[col].nunique(dropna=True)
    if n_unique < 2:
        return None
    if n_unique <= 10:
        # One-Hot
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
        return dummies
    else:
        # 頻度エンコーディング
        freq = df[col].value_counts(normalize=True)
        encoded = df[col].map(freq).fillna(0).astype(float)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(encoded.values.reshape(-1, 1))
        return pd.DataFrame(scaled, columns=[col], index=df.index)


# ============================================================
# 3. カラム組み合わせの生成
# ============================================================
def generate_column_sets(encoded: pd.DataFrame) -> dict:
    sets = {}

    # all
    sets["all"] = encoded.copy()

    # drop_corr: 相関係数 > 0.95 の片方を除外
    corr_matrix = encoded.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    if to_drop:
        sets["drop_corr"] = encoded.drop(columns=to_drop)
    else:
        sets["drop_corr"] = encoded.copy()

    # pca_95: 分散95%保持
    if encoded.shape[1] >= 2:
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE, svd_solver="full")
        pca_data = pca.fit_transform(encoded.values)
        pca_cols = [f"PC{i+1}" for i in range(pca_data.shape[1])]
        sets["pca_95"] = pd.DataFrame(pca_data, columns=pca_cols, index=encoded.index)

    return sets


# ============================================================
# 4. クラスタリング探索
# ============================================================
def run_exploration(column_sets: dict) -> list:
    results = []

    for cs_name, data_df in column_sets.items():
        X = data_df.values
        n_samples = X.shape[0]

        # --- KMeans ---
        max_k = min(10, n_samples - 1)
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
            sil = silhouette_score(X, labels)
            results.append({
                "method": f"KMeans(k={k})",
                "column_set": cs_name,
                "n_clusters": k,
                "silhouette": round(sil, 4),
                "noise_ratio": None,
                "labels": labels,
            })

        # --- Agglomerative ---
        for linkage in ["ward", "complete", "average"]:
            for k in range(2, max_k + 1):
                labels = AgglomerativeClustering(
                    n_clusters=k, linkage=linkage
                ).fit_predict(X)
                sil = silhouette_score(X, labels)
                results.append({
                    "method": f"Agglomerative({linkage},k={k})",
                    "column_set": cs_name,
                    "n_clusters": k,
                    "silhouette": round(sil, 4),
                    "noise_ratio": None,
                    "labels": labels,
                })

        # --- DBSCAN ---
        eps_values = np.linspace(0.3, 2.0, 5)
        min_samples_values = [3, 5, 7, 10]
        for eps in eps_values:
            for ms in min_samples_values:
                labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                n_clusters = len(set(labels) - {-1})
                if n_clusters < 2:
                    continue
                mask = labels != -1
                if mask.sum() < 2:
                    continue
                sil = silhouette_score(X[mask], labels[mask])
                noise_ratio = round((labels == -1).sum() / len(labels), 4)
                results.append({
                    "method": f"DBSCAN(eps={eps:.2f},ms={ms})",
                    "column_set": cs_name,
                    "n_clusters": n_clusters,
                    "silhouette": round(sil, 4),
                    "noise_ratio": noise_ratio,
                    "labels": labels,
                })

    results.sort(key=lambda r: r["silhouette"], reverse=True)
    return results


# ============================================================
# 5. 出力
# ============================================================

# 5-1. コンソール出力
def print_results(results: list, top_n: int = 10):
    print(f"\n[探索完了] {len(results)} パターンを試行\n")
    print(f"Top {min(top_n, len(results))} パターン（Silhouette係数 降順）")
    print("─" * 72)
    header = f"{'Rank':<6}{'手法':<32}{'カラムセット':<14}{'クラスタ数':<10}{'Silhouette':<10}"
    print(header)
    print("─" * 72)
    for i, r in enumerate(results[:top_n]):
        sil_str = f"{r['silhouette']:.4f}"
        if r["silhouette"] < 0:
            sil_str += " ⚠"
        print(f"{i+1:<6}{r['method']:<32}{r['column_set']:<14}{r['n_clusters']:<10}{sil_str:<10}")
    print("─" * 72)

    # 低品質警告
    low = [r for r in results[:top_n] if r["silhouette"] < 0]
    if low:
        print(f"\n⚠ 警告: Silhouette係数 < 0 のパターンが {len(low)} 件あります（低品質）")


# 5-2. 可視化
def save_silhouette_ranking(results: list, output_dir: Path):
    top20 = results[:20]
    labels = [f"{r['method']}\n({r['column_set']})" for r in top20]
    values = [r["silhouette"] for r in top20]
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(top20))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Silhouette Score")
    ax.set_title("Silhouette Ranking (Top 20)")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "silhouette_ranking.png", dpi=150)
    plt.close(fig)


def save_scatter(encoded: pd.DataFrame, labels: np.ndarray, method_name: str,
                 rank: int, output_dir: Path):
    X = encoded.values
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    if X_valid.shape[0] < 2:
        return

    # PCA
    if X_valid.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords_pca = pca.fit_transform(X_valid)
        _plot_scatter(coords_pca, labels_valid, method_name, "PCA",
                      output_dir / f"scatter_top{rank}_pca.png")

    # t-SNE
    perplexity = min(30, X_valid.shape[0] - 1)
    if perplexity >= 2:
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                     random_state=RANDOM_STATE)
        coords_tsne = tsne.fit_transform(X_valid)
        _plot_scatter(coords_tsne, labels_valid, method_name, "t-SNE",
                      output_dir / f"scatter_top{rank}_tsne.png")


def _plot_scatter(coords: np.ndarray, labels: np.ndarray, method: str,
                  proj: str, filepath: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", len(unique_labels))
    for idx, cl in enumerate(unique_labels):
        mask = labels == cl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[idx]],
                   label=f"Cluster {cl}", s=20, alpha=0.7)
    ax.set_title(f"{method} — {proj} projection")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(f"{proj}1")
    ax.set_ylabel(f"{proj}2")
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def save_cluster_profile(encoded: pd.DataFrame, labels: np.ndarray,
                         method_name: str, output_dir: Path):
    """各クラスタの特徴量平均をレーダーチャートで描画。"""
    mask = labels != -1
    df = encoded.iloc[mask].copy()
    df["cluster"] = labels[mask]
    cluster_means = df.groupby("cluster").mean()

    features = list(cluster_means.columns)
    n_features = len(features)
    if n_features < 3:
        return  # レーダーチャートには最低3軸必要
    if n_features > 12:
        # 特徴量が多すぎる場合、分散の大きいものを上位12に絞る
        var_order = df[features].var().sort_values(ascending=False)
        features = list(var_order.head(12).index)
        cluster_means = cluster_means[features]
        n_features = len(features)

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", len(cluster_means))

    for idx, (cl, row) in enumerate(cluster_means.iterrows()):
        values = row.tolist() + [row.iloc[0]]
        ax.plot(angles, values, "o-", linewidth=1.5, label=f"Cluster {cl}",
                color=palette[idx])
        ax.fill(angles, values, alpha=0.1, color=palette[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=7)
    ax.set_title(f"Cluster Profile — {method_name}", y=1.08)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(output_dir / "cluster_profile.png", dpi=150)
    plt.close(fig)


# 5-3. CSV 出力
def save_csv_results(results: list, output_dir: Path):
    rows = []
    for r in results:
        rows.append({
            "method": r["method"],
            "column_set": r["column_set"],
            "n_clusters": r["n_clusters"],
            "silhouette": r["silhouette"],
            "noise_ratio": r["noise_ratio"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "results_all.csv", index=False)


def save_top_labels(df_original: pd.DataFrame, labels: np.ndarray, output_dir: Path):
    out = df_original.copy()
    out["cluster"] = labels
    out.to_csv(output_dir / "cluster_labels_top1.csv", index=False)


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="自動クラスタリング探索ツール")
    parser.add_argument("filepath", help="入力ファイル (CSV / Excel)")
    parser.add_argument("--top", type=int, default=3,
                        help="上位何位まで散布図を出力するか (デフォルト: 3)")
    parser.add_argument("--output", type=str, default="output",
                        help="出力ディレクトリ (デフォルト: output)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. ファイル読み込み
    print(f"[読み込み] {args.filepath}")
    df = load_data(args.filepath)
    print(f"  行数: {len(df)}, カラム数: {len(df.columns)}")

    # 2. 前処理
    print("[前処理] カラム型の自動判定...")
    col_types = detect_column_types(df)
    for col, ctype in col_types.items():
        print(f"  {col}: {ctype}")

    print("[前処理] 自動除外判定...")
    excluded = columns_to_exclude(df, col_types)
    if excluded:
        print(f"  除外カラム: {excluded}")
    else:
        print("  除外カラムなし")

    print("[前処理] エンコーディング...")
    encoded = encode_columns(df, col_types, excluded)
    print(f"  特徴量数: {encoded.shape[1]}")

    # 3. カラム組み合わせ生成
    print("[探索] カラム組み合わせの生成...")
    column_sets = generate_column_sets(encoded)
    print(f"  パターン: {list(column_sets.keys())}")

    # 4. クラスタリング探索
    print("[探索] クラスタリング実行中...")
    results = run_exploration(column_sets)

    if not results:
        sys.exit("[エラー] 有効なクラスタリング結果がありませんでした。")

    # 5. 出力
    print_results(results)

    # 可視化
    print("[出力] 可視化画像を生成中...")
    save_silhouette_ranking(results, output_dir)

    top_n = min(args.top, len(results))
    for rank in range(1, top_n + 1):
        r = results[rank - 1]
        cs_name = r["column_set"]
        save_scatter(column_sets[cs_name], r["labels"], r["method"], rank, output_dir)

    # レーダーチャート（上位1位）
    best = results[0]
    save_cluster_profile(column_sets[best["column_set"]], best["labels"],
                         best["method"], output_dir)

    # CSV
    print("[出力] CSV出力中...")
    save_csv_results(results, output_dir)
    save_top_labels(df, results[0]["labels"], output_dir)

    print(f"\n[完了] 結果は {output_dir}/ に出力されました。")


if __name__ == "__main__":
    main()
