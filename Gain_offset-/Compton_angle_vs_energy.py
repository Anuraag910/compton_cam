import numpy as np
import matplotlib.pyplot as plt

# Constants
m_e = 511  # electron rest mass in keV

# Function to calculate scattered energy
def scattered_energy(E, theta):
    theta_rad = np.radians(theta)  # Convert angle to radians
    return E / (1 + (E / m_e) * (1 - np.cos(theta_rad)))

# Range of scattering angles (0° to 180°)
angles = np.linspace(0, 180, 500)

# Photon source energies in keV
photon_energies = [30, 59.6, 81, 105, 356]

# Plotting
plt.figure(figsize=(10, 6))

for E in photon_energies:
    scattered_energies = scattered_energy(E, angles)
    plt.plot(scattered_energies, angles,  label=f'{E} keV')

# Customize plot
plt.title('Scattered Energy vs Scattering Angle for Different Photon Energies')
plt.ylabel('Scattering Angle (degrees)')
plt.xlabel('Scattered Photon Energy (keV)')
plt.legend(title='Incident Photon Energy')
plt.grid(True)

# Show plot
plt.show()
