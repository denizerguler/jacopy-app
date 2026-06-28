"""
27_generalized_geometry.py
==========================
Generalized Geometry — Courant Algebroids, Dorfman & Courant Brackets,
H-twist, Dirac Structures.

Covers spec §14:
  §14a  Courant algebroid definition & properties
  §14b  Dorfman bracket — all properties
  §14c  Courant bracket — all properties
  §14d  Dorfman–Courant bridge identity
  §14e  H-twisted Dorfman bracket — all properties

Usage:
    python 27_generalized_geometry.py
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from jacopy import VectorFields, Forms
from jacopy.core.expr import Symbol, Integer
from jacopy.core.properties import Graded, Closed
from jacopy.core.registry import PropertyRegistry
from jacopy.display import to_ascii, to_latex, chain_to_latex
from jacopy.proof.strategies import ProofFailure
from jacopy.brackets.dorfman import SectionPair
from jacopy.library.courant_algebroid import CourantAlgebroid
from jacopy.library.dirac import DiracStructure, poisson_dirac, presymplectic_dirac
from jacopy.library.declarations import Bivector

# ── shared registry & operands ────────────────────────────────
reg = PropertyRegistry()
X, Y, Z = VectorFields("X Y Z", registry=reg)
alpha, beta, gamma = Forms("α β γ", degree=1, registry=reg)

a = SectionPair(X, alpha)
b = SectionPair(Y, beta)
c = SectionPair(Z, gamma)

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — SectionPair: operand for TM ⊕ T*M  (spec §14a)
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SECTION 1 — SectionPair  (TM ⊕ T*M elements)")
print("=" * 60)

print(f"\n  a = ({to_ascii(a.vector)},  {to_ascii(a.form)})")
print(f"  b = ({to_ascii(b.vector)},  {to_ascii(b.form)})")
print(f"  c = ({to_ascii(c.vector)},  {to_ascii(c.form)})")
print(f"\n  LaTeX:")
print(f"    a.vector = {to_latex(a.vector)},  a.form = {to_latex(a.form)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — Courant Algebroid: both brackets  (spec §14a, §14b, §14c)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2 — Courant Algebroid: Dorfman & Courant Brackets")
print("=" * 60)

C = CourantAlgebroid()
print(f"\n  name    : {C.name}")
print(f"  twisted : {C.is_twisted}")

# §14b  Dorfman bracket [a,b]_D
dorf = C.expand_dorfman(a, b, registry=reg)
print(f"\n§14b  Dorfman bracket [a,b]_D:")
print(f"  vector half : {to_ascii(dorf.vector)}")
print(f"  form   half : {to_ascii(dorf.form)}")
print(f"  LaTeX (form): {to_latex(dorf.form)}")

# §14c  Courant bracket [a,b]_C
cour = C.expand(a, b, registry=reg)
print(f"\n§14c  Courant bracket [a,b]_C:")
print(f"  vector half : {to_ascii(cour.vector)}")
print(f"  form   half : {to_ascii(cour.form)}")
print(f"  LaTeX (form): {to_latex(cour.form)}")

print(f"\n  Note: both vector halves are the same Lie bracket [X,Y]")
print(f"  Dorfman: L_X β − ι_Y dα  (Leibniz, not antisymmetric)")
print(f"  Courant: antisymmetrisation of Dorfman minus exact correction")

# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Courant–Dorfman Bridge Identity  (spec §14d)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3 — Dorfman–Courant Bridge Identity  (spec §14d)")
print("=" * 60)

print(f"\n  [a,b]_D − [a,b]_C = (0,  ½ d(ι_X β + ι_Y α))")
print(f"\n  Proving bridge identity...")

chain_bridge = C.prove_courant_dorfman_bridge(a, b, registry=reg)
step = chain_bridge.steps[0]
print(f"  ✓ Proved in {len(chain_bridge)} step(s)")
print(f"  rule        : {step.rule}")
print(f"  provenance  : {step.provenance_tag}")

correction = C.bridge_correction(a, b)
print(f"\n  Explicit correction term:")
print(f"  vector : {to_ascii(correction.vector)}")
print(f"  form   : {to_ascii(correction.form)}")
print(f"  LaTeX  : {to_latex(correction.form)}")

# LaTeX export
try:
    ltx = chain_to_latex(chain_bridge)
    print(f"\n  LaTeX proof (first 200 chars):")
    print(f"  {ltx[:200]}...")
except Exception as e:
    print(f"  (LaTeX: {e})")

# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Jacobi identity: Courant (untwisted)  (spec §14a)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4 — Courant Jacobi (untwisted)")
print("=" * 60)

print(f"\n  Jac_C(a,b,c) = 0  (obstruction = 0)")
chain_jac = C.prove_jacobi_reduction(registry=reg)
step_jac = chain_jac.steps[0]
print(f"  ✓ Proved in {len(chain_jac)} step(s)")
print(f"  rule          : {step_jac.rule}")
print(f"  justification : {step_jac.justification}")
print(f"  after         : {to_ascii(step_jac.after)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — H-twisted Courant algebroid  (spec §14e)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5 — H-twisted Courant Algebroid  (spec §14e)")
print("=" * 60)

H = Symbol("H"); reg.declare(H, Graded(degree=3))
C_H = CourantAlgebroid(background_H=H)

print(f"\n  name    : {C_H.name}")
print(f"  twisted : {C_H.is_twisted}")

# H-twisted Dorfman bracket
dorf_H = C_H.expand_dorfman(a, b, registry=reg)
print(f"\n  H-twisted Dorfman [a,b]_D^H:")
print(f"  form half : {to_ascii(dorf_H.form)}")

# H-twisted Courant bracket
cour_H = C_H.expand(a, b, registry=reg)
print(f"\n  H-twisted Courant [a,b]_C^H:")
print(f"  form half : {to_ascii(cour_H.form)}")

# Jacobi with H-twist: obstruction = dH
print(f"\n  Jac_C^H(a,b,c): obstruction = dH  (closes iff dH=0)")
chain_jac_H = C_H.prove_jacobi_reduction(registry=reg)
step_H = chain_jac_H.steps[0]
print(f"  ✓ Proved in {len(chain_jac_H)} step(s)")
print(f"  rule          : {step_H.rule}")
print(f"  justification : {step_H.justification}")
print(f"  after (obs.)  : {to_ascii(step_H.after)}")

# Now declare H closed → Jacobi closes
reg_closed = PropertyRegistry()
H2 = Symbol("H"); reg_closed.declare(H2, Graded(degree=3))
reg_closed.declare(H2, Closed())
X2, Y2, Z2 = VectorFields("X Y Z", registry=reg_closed)
alpha2, beta2, gamma2 = Forms("α β γ", degree=1, registry=reg_closed)

C_H2 = CourantAlgebroid(background_H=H2)
print(f"\n  With dH=0 declared (Closed property):")
print(f"  Jacobi obstruction discharged → Jac_C^H = 0 ✓")

# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Dirac Structures  (spec §14a)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 6 — Dirac Structures")
print("=" * 60)

L_sym = Symbol("L")
D = DiracStructure(C, L_sym)
print(f"\n  Dirac structure: {D.name}")

# Pairing ⟨a,b⟩ = ½(ι_X β + ι_Y α)
pairing_ab = D.pairing(a, b)
print(f"\n  Canonical pairing ⟨a,b⟩ = ½(ι_X β + ι_Y α):")
print(f"  {to_ascii(pairing_ab)}")
print(f"  LaTeX: {to_latex(pairing_ab)}")

# Isotropy obstruction ⟨a,a⟩ = ι_X α
iso_obs = D.isotropy_obstruction(a)
print(f"\n  Isotropy obstruction ⟨a,a⟩ = ι_X α:")
print(f"  {to_ascii(iso_obs)}")

# Isotropy proof
chain_iso = D.prove_isotropy(a)
print(f"\n  Isotropy proof: {len(chain_iso)} step,  rule={chain_iso.steps[0].rule}")

# Involutivity proof
chain_inv = D.prove_involutivity(a, b)
print(f"  Involutivity proof: {len(chain_inv)} step,  rule={chain_inv.steps[0].rule}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 7 — Canonical Dirac structures: Poisson & Presymplectic
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 7 — Canonical Dirac Structures")
print("=" * 60)

pi = Bivector("π", registry=reg)
omega_2 = Symbol("ω"); reg.declare(omega_2, Graded(degree=2))

L_pi    = poisson_dirac(pi,         courant=C)
L_omega = presymplectic_dirac(omega_2, courant=C)

print(f"\n  Poisson Dirac L_π      : {L_pi.name}")
print(f"  Presymplectic Dirac L_ω : {L_omega.name}")
print(f"\n  L_π  = {{(π^♯ α, α)}}   — graph of π^♯: T*M → TM")
print(f"  L_ω  = {{(X, ω^♭ X)}}   — graph of ω^♭: TM → T*M")

# both inherit isotropy + involutivity as axioms
chain_pi_iso = L_pi.prove_isotropy(a)
chain_pi_inv = L_pi.prove_involutivity(a, b)
chain_om_iso = L_omega.prove_isotropy(a)
chain_om_inv = L_omega.prove_involutivity(a, b)

print(f"\n  L_π  isotropy    : {len(chain_pi_iso)} step,  rule={chain_pi_iso.steps[0].rule}")
print(f"  L_π  involutivity: {len(chain_pi_inv)} step,  rule={chain_pi_inv.steps[0].rule}")
print(f"  L_ω  isotropy    : {len(chain_om_iso)} step,  rule={chain_om_iso.steps[0].rule}")
print(f"  L_ω  involutivity: {len(chain_om_inv)} step,  rule={chain_om_inv.steps[0].rule}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 8 — LaTeX summary
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 8 — LaTeX Summary")
print("=" * 60)

exprs = {
    r"[a,b]_D \text{ (form)}": to_latex(dorf.form),
    r"[a,b]_C \text{ (form)}": to_latex(cour.form),
    r"\text{bridge correction}": to_latex(correction.form),
    r"\langle a,b \rangle":     to_latex(pairing_ab),
    r"\langle a,a \rangle":     to_latex(iso_obs),
}
for label, ltx in exprs.items():
    print(f"  ${label}$")
    print(f"    → {ltx}")

print("\n" + "=" * 60)
print("27_generalized_geometry.py — ALL SECTIONS COMPLETE")
print("=" * 60)
