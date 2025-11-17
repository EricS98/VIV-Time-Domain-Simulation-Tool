# common/file_picker.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, Tuple

def _has_tk() -> bool:
    try:
        import tkinter  # noqa
        return True
    except Exception:
        return False

def _mk_root(topmost: bool = True):
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    if topmost:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
    return root

def pick_directory(title: str = "Choose output folder",
                   initialdir: Optional[Path] = None) -> Optional[Path]:
    """Folder picker. Returns Path or None on cancel."""
    if not _has_tk():
        raw = input("No GUI available. Enter output folder path (blank to cancel): ").strip()
        return Path(raw).expanduser() if raw else None

    from tkinter import filedialog
    root = _mk_root()
    try:
        path = filedialog.askdirectory(
            title=title,
            initialdir=str(initialdir) if initialdir else None,
            mustexist=True,
        )
        return Path(path) if path else None
    finally:
        root.destroy()

def pick_file(title: str = "Select a file",
              initialdir: Optional[Path] = None,
              filetypes: Iterable[Tuple[str, str]] = (("All files", "*.*"),)
              ) -> Optional[Path]:
    """Open-file picker. Returns Path or None on cancel."""
    if not _has_tk():
        raw = input("No GUI available. Enter file path (blank to cancel): ").strip()
        return Path(raw).expanduser() if raw else None

    from tkinter import filedialog
    root = _mk_root()
    try:
        path = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir) if initialdir else None,
            filetypes=list(filetypes),
        )
        return Path(path) if path else None
    finally:
        root.destroy()

def pick_save_file(title: str = "Save as…",
                   defaultextension: str = ".png",
                   initialdir: Optional[Path] = None,
                   initialfile: Optional[str] = None,
                   filetypes: Iterable[Tuple[str, str]] = (
                       ("PNG image", "*.png"), ("All files", "*.*"))
                   ) -> Optional[Path]:
    """Save-as picker. Returns Path or None on cancel.
       Use 'initialfile' to suggest a filename."""
    if not _has_tk():
        raw = input(f"No GUI. Enter save path (include {defaultextension}) or blank to cancel: ").strip()
        return Path(raw).expanduser() if raw else None

    from tkinter import filedialog
    root = _mk_root()
    try:
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=defaultextension,
            initialdir=str(initialdir) if initialdir else None,
            initialfile=initialfile,
            filetypes=list(filetypes),
        )
        return Path(path) if path else None
    finally:
        root.destroy()

def pick_folder(title: str = "Choose output folder",
                start_dir: Optional[Path] = None) -> Optional[Path]:
    """Alias for older code."""
    return pick_directory(title=title, initialdir=start_dir)

def choose_output_folder(start_dir: Optional[Path] = None) -> Optional[Path]:
    """Alias for older code."""
    return pick_directory(initialdir=start_dir)