"""Windowsデスクトップショートカット作成スクリプト。

AI Trader 株式拡張のAPIサーバー起動ショートカットを
Windowsデスクトップに作成する。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def create_shortcut() -> None:
    """デスクトップにAI Traderショートカットを作成する。"""
    # プロジェクトルートを特定
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.resolve()

    # Pythonインタープリターのパスを取得
    python_exe = sys.executable

    # デスクトップのパスを取得
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        # OneDriveデスクトップにフォールバック
        desktop = Path.home() / "OneDrive" / "デスクトップ"
    if not desktop.exists():
        desktop = Path.home() / "OneDrive" / "Desktop"
    if not desktop.exists():
        print(f"デスクトップが見つかりません。手動で確認してください: {Path.home()}")
        desktop = Path.home()

    shortcut_path = desktop / "AI Trader 株式拡張.lnk"

    try:
        import winshell  # type: ignore
        from win32com.client import Dispatch  # type: ignore
    except ImportError:
        print("winshell/pywin32が未インストールです。以下のコマンドでインストールしてください:")
        print("  pip install winshell pywin32")
        print()
        # 代替: バッチファイルで作成
        _create_batch_shortcut(project_root, python_exe, desktop)
        return

    # WScriptでショートカット作成
    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.Targetpath = python_exe
    shortcut.Arguments = f'-m uvicorn src.api.server:app --host 0.0.0.0 --port 8765'
    shortcut.WorkingDirectory = str(project_root)
    shortcut.IconLocation = python_exe
    shortcut.Description = "AI Trader 株式拡張 APIサーバーを起動"
    shortcut.save()

    print(f"ショートカットを作成しました: {shortcut_path}")
    print(f"  対象: {python_exe}")
    print(f"  作業ディレクトリ: {project_root}")


def _create_batch_shortcut(
    project_root: Path,
    python_exe: str,
    desktop: Path,
) -> None:
    """バッチファイル形式でショートカットを作成する（winshell不使用の代替）。"""
    batch_path = desktop / "AI_Trader_Start.bat"
    content = f"""@echo off
title AI Trader 株式拡張
cd /d "{project_root}"
echo AI Trader APIサーバーを起動しています...
echo ブラウザで http://localhost:8765/docs にアクセスしてください
"{python_exe}" -m uvicorn src.api.server:app --host 0.0.0.0 --port 8765
pause
"""
    batch_path.write_text(content, encoding="utf-8")
    print(f"バッチファイルを作成しました: {batch_path}")
    print("  ダブルクリックでAPIサーバーが起動します")


def main() -> None:
    """エントリーポイント。"""
    if sys.platform != "win32":
        print(f"このスクリプトはWindows専用です（現在: {sys.platform}）")
        sys.exit(1)

    print("=" * 50)
    print("AI Trader 株式拡張 — ショートカット作成")
    print("=" * 50)

    create_shortcut()

    print()
    print("完了！デスクトップのショートカットをダブルクリックして")
    print("APIサーバーを起動してください。")
    print("ブラウザで http://localhost:8765/docs を開くと")
    print("APIドキュメントを確認できます。")


if __name__ == "__main__":
    main()
