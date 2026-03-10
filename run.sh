#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── .env 読み込み ──
if [ ! -f .env ]; then
    echo "[エラー] .env が見つかりません。.env.example を参考に作成してください。"
    exit 1
fi
# shellcheck disable=SC2046
export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs)

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.11}"

# ── pyenv の初期化 ──
if command -v pyenv &>/dev/null; then
    eval "$(pyenv init -)"
else
    echo "[エラー] pyenv が見つかりません。先にインストールしてください。"
    echo "  https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# ── Python バージョンの確認・インストール ──
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo "[セットアップ] Python $PYTHON_VERSION をインストール中..."
    pyenv install "$PYTHON_VERSION"
fi

# ── 仮想環境の作成 ──
if [ ! -d "$VENV_DIR" ]; then
    echo "[セットアップ] 仮想環境を作成中: $VENV_DIR (Python $PYTHON_VERSION)"
    "$(pyenv prefix "$PYTHON_VERSION")/bin/python" -m venv "$VENV_DIR"
fi

# ── 仮想環境の有効化 ──
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 依存ライブラリのインストール ──
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ── in/ フォルダの確認 ──
INPUT_DIR="in"
if [ ! -d "$INPUT_DIR" ]; then
    echo "[エラー] in/ フォルダが見つかりません。入力データを in/ に配置してください。"
    exit 1
fi

# 入力ファイルを検出（CSV / Excel）
INPUT_FILES=()
while IFS= read -r -d '' f; do
    INPUT_FILES+=("$f")
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.csv" -o -name "*.xlsx" -o -name "*.xls" \) -print0 | sort -z)

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "[エラー] in/ にCSVまたはExcelファイルがありません。"
    exit 1
fi

# ── 実行 ──
echo "========================================"
echo " 自動クラスタリング探索ツール"
echo "========================================"
echo ""

for filepath in "${INPUT_FILES[@]}"; do
    filename="$(basename "$filepath")"
    name="${filename%.*}"
    output_dir="out/$name"

    echo "──── 処理中: $filename → $output_dir/ ────"
    python clustering_explorer.py "$filepath" --output "$output_dir" "$@"
    echo ""
done

echo "========================================"
echo " 全ファイルの処理が完了しました"
echo "========================================"
