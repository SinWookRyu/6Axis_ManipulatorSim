"""
6-DOF Manipulator Simulator — entry point.

Usage:
    python main.py
"""

import sys
import os

# Ensure project root is on sys.path so sub-packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from kinematics import DHParameters
from gui import ManipulatorApp


def main():
    dh_params = DHParameters.fanuc_m10ia()

    root = tk.Tk()
    root.geometry("1280x800")

    app = ManipulatorApp(root, dh_params)

    root.mainloop()


if __name__ == "__main__":
    main()
