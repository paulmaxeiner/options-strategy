import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

np.random.seed(42)

# ----------------------------------------------------
# 1. Generate sample data (synthetic, but realistic-looking)
# ----------------------------------------------------
# Moneyness = Strike / Spot (0.7 = deep OTM put side, 1.3 = deep OTM call side)
moneyness = np.linspace(0.7, 1.3, 20)
# Time to maturity, in years
maturities = np.linspace(0.05, 2.0, 20)

M, T = np.meshgrid(moneyness, maturities)

# Build a synthetic IV surface with the usual stylized features:
#   - a smile: IV rises away from at-the-money (moneyness = 1.0)
#   - a skew: puts trade at higher IV than calls (equity-like skew)
#   - a term structure: short-dated options carry an extra vol premium
atm_vol = 0.18
smile = 0.15 * (M - 1.0) ** 2
skew = -0.08 * (M - 1.0)
term_structure = 0.05 * np.exp(-1.5 * T)
noise = np.random.normal(scale=0.003, size=M.shape)

IV = atm_vol + smile + skew + term_structure + noise
IV = np.clip(IV, 0.05, None)  # keep vols positive/sane

# ----------------------------------------------------
# 2. Plot the surface using the "terrain" colormap
# ----------------------------------------------------
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    M, T, IV,
    cmap="terrain",
    edgecolor="none",
    antialiased=True,
    alpha=0.95,
)

ax.set_xlabel("Moneyness (K / Spot)")
ax.set_ylabel("Time to Maturity (yrs)")
ax.set_zlabel("Implied Volatility")
ax.set_title("Sample Implied Volatility Surface")
ax.view_init(elev=25, azim=45)

cbar = fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)
cbar.set_label("Implied Volatility")

plt.tight_layout()
plt.savefig("iv_surface.png", dpi=150)
plt.show()