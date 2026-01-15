#!/usr/bin/env python3
"""
VERITAS GUI - Simple drag-and-drop interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import sys
import os
from pathlib import Path

class VeritasGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VERITAS - Deepfake Detection")
        self.root.geometry("600x500")
        self.root.configure(bg='#1a1a2e')

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(
            self.root,
            text="PROJECT VERITAS",
            font=("Helvetica", 24, "bold"),
            fg="#00ffff",
            bg="#1a1a2e"
        )
        title.pack(pady=20)

        subtitle = tk.Label(
            self.root,
            text="Forensic Deepfake Detection",
            font=("Helvetica", 12),
            fg="#888888",
            bg="#1a1a2e"
        )
        subtitle.pack()

        # Drop zone frame
        self.drop_frame = tk.Frame(
            self.root,
            bg="#2a2a4e",
            highlightbackground="#00ffff",
            highlightthickness=2,
            width=500,
            height=150
        )
        self.drop_frame.pack(pady=30)
        self.drop_frame.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="Click to select video\nor paste YouTube URL below",
            font=("Helvetica", 14),
            fg="#cccccc",
            bg="#2a2a4e",
            cursor="hand2"
        )
        self.drop_label.pack(expand=True)
        self.drop_label.bind("<Button-1>", self.browse_file)
        self.drop_frame.bind("<Button-1>", self.browse_file)

        # URL entry
        url_frame = tk.Frame(self.root, bg="#1a1a2e")
        url_frame.pack(pady=10)

        tk.Label(
            url_frame,
            text="Or paste URL:",
            fg="#888888",
            bg="#1a1a2e"
        ).pack(side=tk.LEFT, padx=5)

        self.url_entry = tk.Entry(url_frame, width=50)
        self.url_entry.pack(side=tk.LEFT, padx=5)

        # Analyze button
        self.analyze_btn = tk.Button(
            self.root,
            text="ANALYZE",
            font=("Helvetica", 14, "bold"),
            fg="#1a1a2e",
            bg="#00ffff",
            activebackground="#00cccc",
            width=20,
            height=2,
            command=self.start_analysis
        )
        self.analyze_btn.pack(pady=20)

        # Status
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#1a1a2e"
        )
        self.status_label.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=10)

        # Result frame (hidden initially)
        self.result_frame = tk.Frame(self.root, bg="#1a1a2e")

        self.selected_file = None

    def browse_file(self, event=None):
        filepath = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.selected_file = filepath
            filename = Path(filepath).name
            self.drop_label.config(text=f"Selected:\n{filename}")
            self.url_entry.delete(0, tk.END)

    def start_analysis(self):
        target = self.url_entry.get().strip() or self.selected_file

        if not target:
            messagebox.showwarning("No Input", "Please select a video file or enter a URL")
            return

        self.analyze_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Analyzing...", fg="#ffff00")
        self.progress.start(10)

        # Run in thread
        thread = threading.Thread(target=self.run_analysis, args=(target,))
        thread.start()

    def run_analysis(self, target):
        try:
            script_dir = Path(__file__).parent
            venv_python = script_dir / ".venv" / "bin" / "python"
            veritas_script = script_dir / "veritas.py"

            result = subprocess.run(
                [str(venv_python), str(veritas_script), target],
                capture_output=True,
                text=True,
                cwd=str(script_dir)
            )

            self.root.after(0, self.analysis_complete, result)

        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))

    def analysis_complete(self, result):
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)

        if result.returncode == 0:
            # Parse output for verdict
            output = result.stdout
            if "HIGH CONFIDENCE" in output:
                verdict = "HIGH PROBABILITY DEEPFAKE"
                color = "#ff0000"
            elif "MEDIUM CONFIDENCE" in output:
                verdict = "POSSIBLE MANIPULATION"
                color = "#ffaa00"
            else:
                verdict = "LIKELY AUTHENTIC"
                color = "#00ff00"

            self.status_label.config(text=f"VERDICT: {verdict}", fg=color)

            messagebox.showinfo(
                "Analysis Complete",
                f"Verdict: {verdict}\n\nFiles created:\n- veritas_report.json\n- veritas_output.mp4"
            )
        else:
            self.status_label.config(text="Analysis failed", fg="#ff0000")
            messagebox.showerror("Error", f"Analysis failed:\n{result.stderr[:500]}")

    def analysis_error(self, error):
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Error", fg="#ff0000")
        messagebox.showerror("Error", f"Error: {error}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VeritasGUI()
    app.run()
