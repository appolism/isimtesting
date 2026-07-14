import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
no_preview = True
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, PathSolver, RadioMapSolver, subcarrier_frequencies
print(f"Mitsuba variant: {mi.variant()}")