"""
Unit-conversion constants for energy (eV, keV, MeV, GeV), length (m, cm, mm, um),
energy-to-SI (J to eV), and time (s, ns, ps). Import as `common.units` and
multiply to convert between units, e.g. `energy * U.MeV2eV`.
"""
keV2eV  = 1e3
MeV2eV  = 1e6
GeV2eV  = 1e9
eV2keV  = 1e-3
eV2MeV  = 1e-6
eV2GeV  = 1e-9
MeV2GeV = 1e-3
m2um    = 1e6
m2mm    = 1e3
m2cm    = 1e2
cm2m    = 1e-2
mm2m    = 1e-3
um2m    = 1e-6
cm2um   = 1e4
um2cm   = 1e-4
um2mm   = 1e-3
mm2cm   = 1e-1
cm2mm   = 1e1
J2eV    = 6.242e+18
s2ns    = 1e9
ns2s    = 1e-9
s2ps    = 1e12
ps2s    = 1e-12

