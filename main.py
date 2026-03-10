#!/usr/bin/env python3
"""
異常検知システム — Isolation Forest + SHAP + PCA可視化 (Tableau連携)

in/ フォルダのCSVを読み込み、自動前処理 → Isolation Forest → SHAP → PCA を実行し、
結果を out/ フォルダにCSV一式として出力する。
"""

import logging
import sys
import warnings
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 設定パラメータ
# ============================================================
CONTAMINATION = 0.05          # 異常割合の想定値
N_ESTIMATORS = 200            # Isolation Forest の木の本数
VARIANCE_THRESHOLD = 0.01     # 低分散カラム除外の閾値
CORR_THRESHOLD = 0.95         # 高相関カラム除外の閾値
SHAP_ALL = False              # True: 全件SHAP計算 / False: 異常レコードのみ
PCA_VARIANCE_WARNING = 0.50   # 累積寄与率の警告閾値
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# ============================================================
# ログ設定
# ============================================================
OUTPUT_DIR = Path("out")


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("anomaly_detection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(output_dir / "processing.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ============================================================
# 1. CSV読み込み・結合
# ============================================================
def detect_encoding(filepath: Path) -> str:
    """chardet で文字コードを判定する。"""
    with open(filepath, "rb") as f:
        raw = f.read(100_000)
    result = chardet.detect(raw)
    return result.get("encoding", "utf-8") or "utf-8"


def load_csvs(input_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """in/ 内の全CSVを読み込んで縦結合する。"""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        logger.error("in/ フォルダにCSVファイルがありません。")
        sys.exit(1)

    frames = []
    for f in csv_files:
        enc = detect_encoding(f)
        logger.info(f"読み込み: {f.name} (encoding={enc})")
        df = pd.read_csv(f, encoding=enc)
        logger.info(f"  行数={len(df)}, カラム数={len(df.columns)}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"結合後: 行数={len(combined)}, カラム数={len(combined.columns)}")
    return combined


# ============================================================
# 2. 自動型判定
# ============================================================
def _is_date_column(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return False
    sample = series.dropna().head(50)
    if len(sample) == 0:
        return False
    # 純粋な数値文字列は日付にしない
    try:
        pd.to_numeric(sample)
        return False
    except (ValueError, TypeError):
        pass
    try:
        parsed = pd.to_datetime(sample, format="mixed")
        return parsed.notna().mean() > 0.8
    except (ValueError, TypeError, OverflowError):
        return False


def detect_column_types(df: pd.DataFrame, logger: logging.Logger) -> dict:
    """各カラムの型を判定する。"""
    col_types = {}
    n = len(df)

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            col_types[col] = "unknown"
            continue

        # bool
        if pd.api.types.is_bool_dtype(series):
            col_types[col] = "bool"
            continue

        # 数値
        if pd.api.types.is_numeric_dtype(series):
            col_types[col] = "numeric"
            continue

        # 日付
        if _is_date_column(series):
            col_types[col] = "date"
            continue

        # object: 数値文字列チェック
        try:
            pd.to_numeric(series)
            col_types[col] = "numeric"
            continue
        except (ValueError, TypeError):
            pass

        # カテゴリ (ユニーク数比率で分岐)
        n_unique = series.nunique()
        if n > 0 and n_unique <= n * 0.5:
            col_types[col] = "category"
        else:
            col_types[col] = "category_high_cardinality"

    for col, ctype in col_types.items():
        logger.info(f"  型判定: {col} → {ctype}")

    return col_types


# ============================================================
# 3. 前処理
# ============================================================
def preprocess(df: pd.DataFrame, col_types: dict,
               logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    """欠損補完・エンコーディング・除外を行い、数値特徴量 DataFrame を返す。"""
    excluded = []
    n = len(df)
    frames = []

    for col in df.columns:
        ctype = col_types.get(col, "unknown")

        # --- 欠損率50%超 → 除外 ---
        missing_rate = df[col].isna().sum() / n if n > 0 else 0
        if missing_rate > 0.5:
            excluded.append(col)
            logger.warning(f"  除外(欠損率{missing_rate:.0%}): {col}")
            continue

        if ctype == "unknown":
            excluded.append(col)
            logger.warning(f"  除外(型不明): {col}")
            continue

        # --- bool ---
        if ctype == "bool":
            frames.append(df[col].astype(int).to_frame())
            continue

        # --- 数値 ---
        if ctype == "numeric":
            num = pd.to_numeric(df[col], errors="coerce")
            num = num.fillna(num.median())
            frames.append(num.to_frame())
            continue

        # --- 日付 ---
        if ctype == "date":
            dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
            feat = pd.DataFrame(index=df.index)
            feat[f"{col}_year"] = dt.dt.year.fillna(0)
            feat[f"{col}_month"] = dt.dt.month.fillna(0)
            feat[f"{col}_day"] = dt.dt.day.fillna(0)
            feat[f"{col}_hour"] = dt.dt.hour.fillna(0)
            feat[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0)
            frames.append(feat)
            continue

        # --- カテゴリ (低カーディナリティ) → Label Encoding ---
        if ctype == "category":
            s = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "MISSING")
            le = LabelEncoder()
            encoded = le.fit_transform(s.astype(str))
            frames.append(pd.DataFrame(encoded, columns=[col], index=df.index))
            continue

        # --- 高カーディナリティ → 頻度エンコーディング ---
        if ctype == "category_high_cardinality":
            s = df[col].fillna("MISSING")
            freq = s.value_counts(normalize=True)
            encoded = s.map(freq).astype(float)
            frames.append(encoded.to_frame())
            continue

    if not frames:
        logger.error("有効な特徴量が0件です。データを確認してください。")
        sys.exit(1)

    feature_df = pd.concat(frames, axis=1)
    # 重複カラム名を解消
    if feature_df.columns.duplicated().any():
        cols = feature_df.columns.tolist()
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        feature_df.columns = new_cols

    feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)
    return feature_df, excluded


# ============================================================
# 4. 特徴量選択
# ============================================================
def select_features(feature_df: pd.DataFrame,
                    logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    """低分散除去・高相関除去を行う。"""
    excluded = []

    # 分散フィルタ
    variances = feature_df.var()
    low_var_cols = variances[variances < VARIANCE_THRESHOLD].index.tolist()
    if low_var_cols:
        logger.info(f"  低分散除外 (< {VARIANCE_THRESHOLD}): {low_var_cols}")
        excluded.extend(low_var_cols)
        feature_df = feature_df.drop(columns=low_var_cols)

    # 相関フィルタ
    if feature_df.shape[1] >= 2:
        corr_matrix = feature_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
        if high_corr_cols:
            logger.info(f"  高相関除外 (> {CORR_THRESHOLD}): {high_corr_cols}")
            excluded.extend(high_corr_cols)
            feature_df = feature_df.drop(columns=high_corr_cols)

    if feature_df.shape[1] == 0:
        logger.error("特徴量選択後に有効カラムが0件です。")
        sys.exit(1)

    logger.info(f"  最終特徴量数: {feature_df.shape[1]}")
    return feature_df, excluded


# ============================================================
# 5. スケーリング
# ============================================================
def scale_features(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df.values)
    return pd.DataFrame(scaled, columns=feature_df.columns, index=feature_df.index), scaler


# ============================================================
# 6. Isolation Forest
# ============================================================
def run_isolation_forest(X_scaled: pd.DataFrame,
                         logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, IsolationForest]:
    logger.info(f"Isolation Forest: n_estimators={N_ESTIMATORS}, "
                f"contamination={CONTAMINATION}")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_features=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled.values)

    # スコア: score_samples のマイナス反転（高いほど異常）
    anomaly_score = -model.score_samples(X_scaled.values)
    # ラベル: -1→1(異常), 1→0(正常)
    raw_pred = model.predict(X_scaled.values)
    is_anomaly = (raw_pred == -1).astype(int)

    n_anomaly = is_anomaly.sum()
    logger.info(f"  異常件数: {n_anomaly} / {len(is_anomaly)} "
                f"({n_anomaly / len(is_anomaly):.1%})")
    return anomaly_score, is_anomaly, model


# ============================================================
# 7. SHAP 異常原因分析
# ============================================================
def run_shap(model: IsolationForest, X_scaled: pd.DataFrame,
             is_anomaly: np.ndarray,
             logger: logging.Logger) -> pd.DataFrame | None:
    try:
        logger.info("SHAP 計算中...")
        explainer = shap.TreeExplainer(model)

        if SHAP_ALL:
            shap_values = explainer.shap_values(X_scaled.values)
        else:
            # 異常レコードのみ
            anomaly_mask = is_anomaly == 1
            if anomaly_mask.sum() == 0:
                logger.warning("異常レコードが0件のためSHAP計算をスキップします。")
                return None
            X_anomaly = X_scaled.values[anomaly_mask]
            shap_values_partial = explainer.shap_values(X_anomaly)
            # 全件サイズの配列に埋め込む
            shap_values = np.zeros_like(X_scaled.values)
            shap_values[anomaly_mask] = shap_values_partial

        shap_df = pd.DataFrame(shap_values, columns=X_scaled.columns,
                               index=X_scaled.index)
        logger.info(f"  SHAP 計算完了 (shape={shap_df.shape})")
        return shap_df

    except Exception as e:
        logger.warning(f"SHAP 計算エラー: {e}")
        logger.warning("SHAP 出力をスキップします。")
        return None


# ============================================================
# 8. PCA 2次元圧縮
# ============================================================
def run_pca(X_scaled: pd.DataFrame,
            logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled.values)

    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)
    logger.info(f"  PCA 寄与率: PC1={variance_ratio[0]:.4f}, "
                f"PC2={variance_ratio[1]:.4f}, 累積={cumulative[1]:.4f}")

    if cumulative[1] < PCA_VARIANCE_WARNING:
        logger.warning(
            f"  PCA 累積寄与率が {PCA_VARIANCE_WARNING:.0%} を下回っています "
            f"({cumulative[1]:.1%})。2次元での表現力が低い可能性があります。")

    # 全主成分の寄与率（フル PCA）
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled.values)
    full_variance = pca_full.explained_variance_ratio_

    return coords, full_variance


# ============================================================
# 9. 出力
# ============================================================
def save_outputs(df_original: pd.DataFrame, X_scaled: pd.DataFrame,
                 anomaly_score: np.ndarray, is_anomaly: np.ndarray,
                 shap_df: pd.DataFrame | None, pca_coords: np.ndarray,
                 pca_full_variance: np.ndarray,
                 excluded_columns: list[str],
                 output_dir: Path, logger: logging.Logger):
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- top_feature / top_shap_value ---
    if shap_df is not None:
        abs_shap = shap_df.abs()
        top_feature = abs_shap.idxmax(axis=1)
        top_shap_value = abs_shap.max(axis=1)
    else:
        top_feature = pd.Series(["N/A"] * len(df_original), index=df_original.index)
        top_shap_value = pd.Series([0.0] * len(df_original), index=df_original.index)

    # --- result.csv ---
    result = df_original.copy()
    result["anomaly_score"] = anomaly_score
    result["is_anomaly"] = is_anomaly
    result["top_feature"] = top_feature.values
    result["top_shap_value"] = top_shap_value.values
    result.to_csv(output_dir / "result.csv", index=False, encoding="utf-8-sig")
    logger.info(f"  出力: result.csv ({len(result)} 行)")

    # --- shap_summary.csv ---
    if shap_df is not None:
        shap_df.to_csv(output_dir / "shap_summary.csv", index=False,
                       encoding="utf-8-sig")
        logger.info(f"  出力: shap_summary.csv")

    # --- pca_2d.csv ---
    pca_out = df_original.copy()
    pca_out.insert(0, "PC1", pca_coords[:, 0])
    pca_out.insert(1, "PC2", pca_coords[:, 1])
    pca_out["anomaly_score"] = anomaly_score
    pca_out["is_anomaly"] = is_anomaly
    pca_out["top_feature"] = top_feature.values
    pca_out.to_csv(output_dir / "pca_2d.csv", index=False, encoding="utf-8-sig")
    logger.info(f"  出力: pca_2d.csv")

    # --- pca_variance.csv ---
    pca_var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca_full_variance))],
        "variance_ratio": pca_full_variance,
        "cumulative_ratio": np.cumsum(pca_full_variance),
    })
    pca_var_df.to_csv(output_dir / "pca_variance.csv", index=False,
                      encoding="utf-8-sig")
    logger.info(f"  出力: pca_variance.csv ({len(pca_var_df)} 成分)")

    # --- excluded_columns.txt ---
    with open(output_dir / "excluded_columns.txt", "w", encoding="utf-8") as f:
        for col in excluded_columns:
            f.write(col + "\n")
    logger.info(f"  出力: excluded_columns.txt ({len(excluded_columns)} カラム)")


# ============================================================
# メイン
# ============================================================
def main():
    input_dir = Path("in")
    output_dir = OUTPUT_DIR

    logger = setup_logging(output_dir)
    logger.info("=" * 50)
    logger.info("異常検知システム 開始")
    logger.info("=" * 50)

    # 1. 読み込み
    if not input_dir.exists():
        logger.error("in/ フォルダが存在しません。")
        sys.exit(1)

    df = load_csvs(input_dir, logger)

    if len(df) < 3:
        logger.error("データ行数が少なすぎます（最低3行必要）。")
        sys.exit(1)

    # 2. 型判定
    logger.info("自動型判定...")
    col_types = detect_column_types(df, logger)

    # 3. 前処理
    logger.info("前処理（欠損補完・エンコーディング）...")
    feature_df, excluded_preprocess = preprocess(df, col_types, logger)

    # 4. 特徴量選択
    logger.info("特徴量選択...")
    feature_df, excluded_selection = select_features(feature_df, logger)

    all_excluded = excluded_preprocess + excluded_selection

    # 5. スケーリング
    logger.info("StandardScaler 適用...")
    X_scaled, scaler = scale_features(feature_df)

    # 6. Isolation Forest
    logger.info("Isolation Forest 実行...")
    anomaly_score, is_anomaly, model = run_isolation_forest(X_scaled, logger)

    # 7. SHAP
    logger.info("SHAP 異常原因分析...")
    shap_df = run_shap(model, X_scaled, is_anomaly, logger)

    # 8. PCA
    logger.info("PCA 2次元圧縮...")
    pca_coords, pca_full_variance = run_pca(X_scaled, logger)

    # 9. 出力
    logger.info("結果出力...")
    save_outputs(df, X_scaled, anomaly_score, is_anomaly,
                 shap_df, pca_coords, pca_full_variance,
                 all_excluded, output_dir, logger)

    logger.info("=" * 50)
    logger.info("異常検知システム 完了")
    logger.info(f"結果は {output_dir}/ に出力されました。")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
