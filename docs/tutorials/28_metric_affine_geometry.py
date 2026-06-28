"""
28_metric_affine_geometry.py
============================
Metric-Affine Geometry
Covers spec §11:
  §11b   Connection (nabla)
  §11f   Torsion T(nabla)
  §11g   Curvature R(nabla)
  §11i   Connection 1-forms omega^a_b
  §11j   Torsion 2-forms T^a
  §11k   Curvature 2-forms R^a_b
  §11s   Bianchi identities
  §11t   Cartan structure equations (on a frame)
  §11p   Delta(nabla, nabla') = nabla - nabla'

Usage:
    python 28_metric_affine_geometry.py
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.connection import AffineConnection, koszul_connection
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.torsion_curvature import Torsion, Curvature
from jacopy.calculus.local_frame import LocalFrame
from jacopy.calculus.cartan_forms import ConnectionForm, TorsionForm, CurvatureForm
from jacopy.core.registry import PropertyRegistry
from jacopy.core.expr import Symbol, Sum, Neg
from jacopy.core.properties import Graded
from jacopy.display import to_ascii, to_latex, chain_to_latex
from jacopy.library.bianchi_problem import BianchiProblem
from jacopy.library.cartan_structure import CartanStructureProblem

# ── shared objects ────────────────────────────────────────────
reg = PropertyRegistry()
nabla = AffineConnection("∇")
X = Derivation("X", 0)
Y = Derivation("Y", 0)
Z = Derivation("Z", 0)
W = Derivation("W", 0)

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — AffineConnection & its axioms  (spec §11b)
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SECTION 1 — Affine Connection  (spec §11b)")
print("=" * 60)

nabla_XY = nabla.eval(X, Y)
print(f"\n§11b  Connection: {nabla}")
print(f"      ∇_X Y = {to_ascii(nabla_XY)}")
print(f"      LaTeX : {to_latex(nabla_XY)}")

print(f"\n  Four defining axioms:")
print(f"  1. ∇_{{X+Y}} Z  = ∇_X Z + ∇_Y Z   (X-linearity)")
print(f"  2. ∇_{{fX}} Y  = f ∇_X Y            (X-scalar-pull)")
print(f"  3. ∇_X (Y+Z)   = ∇_X Y + ∇_X Z   (Y-additivity)")
print(f"  4. ∇_X (fY)    = X(f)·Y + f ∇_X Y  (Y-Leibniz)")

# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — Torsion & Curvature  (spec §11f, §11g)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2 — Torsion & Curvature  (spec §11f, §11g)")
print("=" * 60)

T = Torsion(nabla, X, Y)
R = Curvature(nabla, X, Y, W)

print(f"\n§11f  Torsion:")
print(f"      T(X,Y) := ∇_X Y − ∇_Y X − [X,Y]")
print(f"      T(X,Y) = {to_ascii(T)}")
print(f"      LaTeX  : {to_latex(T)}")

print(f"\n§11g  Curvature:")
print(f"      R(X,Y)W := ∇_X ∇_Y W − ∇_Y ∇_X W − ∇_{{[X,Y]}} W")
print(f"      R(X,Y)W = {to_ascii(R)}")
print(f"      LaTeX   : {to_latex(R)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Bianchi identities  (spec §11s)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3 — Bianchi Identities  (spec §11s)")
print("=" * 60)

prob = BianchiProblem(nabla, registry=reg)
print(f"\n  BianchiProblem engine rules: {len(prob.engine.definitions)}")
print(f"  connection: {prob.connection}")

# First Bianchi identity
# cycl_{X,Y,Z} R(X,Y)Z = cycl_{X,Y,Z} [(∇_X T)(Y,Z) + T(T(X,Y), Z)]
print(f"\n§11s  First Bianchi identity:")
print(f"      cycl_{{X,Y,Z}} R(X,Y)Z = cycl_{{X,Y,Z}} [(∇_X T)(Y,Z) + T(T(X,Y),Z)]")
res1 = prob.prove_first_bianchi(X, Y, W)
print(f"      ✓ ok={res1.ok},  lhs steps={len(res1.lhs_steps)},  rhs steps={len(res1.rhs_steps)}")

# Second Bianchi identity
# cycl_{X,Y,Z} (∇_X R)(Y,Z)W = cycl_{X,Y,Z} R(X, T(Y,Z)) W
print(f"\n§11s  Second Bianchi identity:")
print(f"      cycl_{{X,Y,Z}} (∇_X R)(Y,Z)W = cycl_{{X,Y,Z}} R(X, T(Y,Z)) W")
res2 = prob.prove_second_bianchi(X, Y, W, Z)
print(f"      ✓ ok={res2.ok},  lhs steps={len(res2.lhs_steps)},  rhs steps={len(res2.rhs_steps)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Connection forms, Torsion forms, Curvature forms
#             (spec §11i, §11j, §11k)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4 — Connection / Torsion / Curvature Forms  (spec §11i–k)")
print("=" * 60)

F = LocalFrame("F", dim=3)
omega_ab = ConnectionForm(nabla, F, "a", "b")
T_a      = TorsionForm(nabla, F, "a")
R_ab     = CurvatureForm(nabla, F, "a", "b")

print(f"\n§11i  Connection 1-form: ω^a_b(∇) = {omega_ab}")
print(f"      LaTeX: {to_latex(omega_ab)}")

print(f"\n§11j  Torsion 2-form: T^a(∇) = {T_a}")
print(f"      LaTeX: {to_latex(T_a)}")

print(f"\n§11k  Curvature 2-form: R^a_b(∇) = {R_ab}")
print(f"      LaTeX: {to_latex(R_ab)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Cartan structure equations on a frame  (spec §11t)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5 — Cartan Structure Equations on a Frame  (spec §11t)")
print("=" * 60)

print(f"\n  T^a   = de^a   + Σ_b ω^a_b ∧ e^b        (Cartan I)")
print(f"  R^a_b = dω^a_b + Σ_c ω^a_c ∧ ω^c_b      (Cartan II)")

cartan_prob = CartanStructureProblem(nabla, F)
print(f"\n  CartanStructureProblem engine rules: {len(cartan_prob.engine.definitions)}")

U = Derivation("U", 0)
V = Derivation("V", 0)

# LHS / RHS display
lhs1 = cartan_prob.first_cartan_lhs(U, V, "a")
rhs1 = cartan_prob.first_cartan_rhs(U, V, "a")
lhs2 = cartan_prob.second_cartan_lhs(U, V, "a", "b")
rhs2 = cartan_prob.second_cartan_rhs(U, V, "a", "b")

print(f"\n  Cartan I:")
print(f"    LHS : {to_ascii(lhs1)}")
print(f"    RHS : {to_ascii(rhs1)}")

print(f"\n  Cartan II:")
print(f"    LHS : {to_ascii(lhs2)}")
print(f"    RHS : {to_ascii(rhs2)}")

# Prove Cartan I
print(f"\n  Proving Cartan I ...")
res_c1 = cartan_prob.prove_first_cartan(U, V, "a")
print(f"  ✓ ok={res_c1.ok},  steps={len(res_c1.steps)}")

# Prove Cartan II
print(f"\n  Proving Cartan II ...")
res_c2 = cartan_prob.prove_second_cartan(U, V, "a", "b")
print(f"  ✓ ok={res_c2.ok},  steps={len(res_c2.steps)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Delta(nabla, nabla') = nabla - nabla'  (spec §11p)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 6 — Difference of Connections  (spec §11p)")
print("=" * 60)

nabla2 = AffineConnection("∇'")
nabla_XY  = nabla.eval(X, Y)
nabla2_XY = nabla2.eval(X, Y)
delta_XY  = Sum(nabla_XY, Neg(nabla2_XY))

print(f"\n§11p  Δ(∇, ∇')(X, Y) := ∇_X Y − ∇'_X Y")
print(f"      ∇_X Y  = {to_ascii(nabla_XY)}")
print(f"      ∇'_X Y = {to_ascii(nabla2_XY)}")
print(f"      Δ(X,Y) = {to_ascii(delta_XY)}")
print(f"      LaTeX  : {to_latex(delta_XY)}")
print(f"\n      Property: Δ is C^∞(M)-linear in both X and Y")
print(f"      (follows from linearity axioms of both ∇ and ∇')")

# ═════════════════════════════════════════════════════════════════════
# SECTION 7 — Koszul connection on T*M  (algebroid variant)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 7 — Koszul Connection on T*M  (algebroid variant)")
print("=" * 60)

pi_sharp = Anchor("π^♯")
nabla_tilde = koszul_connection("∇̃", anchor=pi_sharp)

print(f"\n  Koszul connection: {nabla_tilde}")
print(f"  anchor           : {nabla_tilde.anchor}")
print(f"  bracket          : {nabla_tilde.bracket}")

T_tilde = Torsion(nabla_tilde, X, Y)
R_tilde = Curvature(nabla_tilde, X, Y, W)
print(f"\n  T̃(X,Y) = {to_ascii(T_tilde)}")
print(f"  R̃(X,Y)W = {to_ascii(R_tilde)}")

print(f"\n  Bianchi identities hold on T*M too (BianchiProblem handles both)")
prob_tilde = BianchiProblem(nabla_tilde, registry=reg)
res_tilde1 = prob_tilde.prove_first_bianchi(X, Y, W)
res_tilde2 = prob_tilde.prove_second_bianchi(X, Y, W, Z)
print(f"  First Bianchi  : ok={res_tilde1.ok},  steps={len(res_tilde1.lhs_steps)}")
print(f"  Second Bianchi : ok={res_tilde2.ok},  steps={len(res_tilde2.lhs_steps)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 8 — LaTeX summary
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 8 — LaTeX Summary")
print("=" * 60)

exprs = {
    r"\nabla_X Y":          to_latex(nabla_XY),
    r"T(X,Y)":              to_latex(T),
    r"R(X,Y)W":             to_latex(R),
    r"\omega^a_{\ b}":      to_latex(omega_ab),
    r"T^a":                 to_latex(T_a),
    r"R^a_{\ b}":           to_latex(R_ab),
    r"\Delta(\nabla,\nabla')(X,Y)": to_latex(delta_XY),
}
for label, ltx in exprs.items():
    print(f"  ${label}$")
    print(f"    → {ltx}")

print("\n" + "=" * 60)
print("28_metric_affine_geometry.py — ALL SECTIONS COMPLETE")
print("=" * 60)
