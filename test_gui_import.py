"""Test that GUI module imports without error (no window opened)."""
import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")   # headless backend for import test

import tkinter as tk
from kinematics import DHParameters
from gui.main_window import ManipulatorApp

print("GUI imports OK")
print("ManipulatorApp:", ManipulatorApp)
