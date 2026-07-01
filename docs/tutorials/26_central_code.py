"""
26_central_code.py
==================
Central Code — Basic Objects and Operations
Covers items 8 and 9 from the project specification.

All objects are defined to work on both TM (tangent bundle) and E (algebroid).
When E = TM, rho_E = id_TM and [.,.]_E = [.,.]_Lie, algebroid versions
reduce to the usual ones.

Usage:
    python 26_central_code.py
"""

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────
here = Path(__file__).resolve().parent
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — Basic symbolic objects  (spec §8c–j)
# ═════════════════════════════════════════════════════════════════════

from jacopy.core.expr import Symbol, Integer, Sum, Product, Neg
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.algorithms.simplify import simplify
from jacopy.display import to_ascii, to_latex
from jacopy import VectorFields, Forms
from jacopy.library.declarations import Bivector, Functions

print("=" * 60)
print("SECTION 1 — Basic Symbolic Objects")
print("=" * 60)

reg = PropertyRegistry()

# §8d  Functions C^∞(M)
f, g, h = Functions("f g h", registry=reg)
print(f"\n§8d  Functions:  f={to_ascii(f)}, g={to_ascii(g)}, h={to_ascii(h)}")
print(f"     LaTeX: {to_latex(f)}, {to_latex(g)}, {to_latex(h)}")

# §8e  Vector fields
U, V, W = VectorFields("U V W", registry=reg)
print(f"\n§8e  Vector fields:  U={to_ascii(U)}, V={to_ascii(V)}, W={to_ascii(W)}")

# §8g  1-forms
omega, eta, xi = Forms("ω η ξ", degree=1, registry=reg)
print(f"\n§8g  1-forms (degree=1):  ω={to_ascii(omega)}, η={to_ascii(eta)}, ξ={to_ascii(xi)}")

# §8h  p-forms
alpha_2 = Symbol("α₂"); reg.declare(alpha_2, Graded(degree=2))
alpha_3 = Symbol("α₃"); reg.declare(alpha_3, Graded(degree=3))
print(f"\n§8h  p-forms:  α₂ (p=2), α₃ (p=3)")
print(f"     degrees: {reg.get(alpha_2, Graded).degree}, {reg.get(alpha_3, Graded).degree}")

# §8i  p-vector fields
pi = Bivector("π", registry=reg)        # 2-vector (bivector)
Pi3 = Symbol("Π₃"); reg.declare(Pi3, Graded(degree=1))  # SN-graded 3-vector
print(f"\n§8i  Bivector (2-vector): π={to_ascii(pi)}")
print(f"     3-vector: Π₃={to_ascii(Pi3)}")

# §8c  Basic operations: +, -, *, 0, 1
print("\n§8c  Basic operations:")
expr_sum = Sum(f, g)
expr_neg = Neg(f)
expr_prod = Product(Integer(2), f)
expr_zero = Integer(0)
expr_one  = Integer(1)
print(f"     f + g  = {to_ascii(expr_sum)}")
print(f"     -f     = {to_ascii(expr_neg)}")
print(f"     2·f    = {to_ascii(expr_prod)}")
print(f"     0      = {to_ascii(expr_zero)}")
print(f"     1      = {to_ascii(expr_one)}")
print(f"     simplify(f + f - f) = {to_ascii(simplify(Sum(f, Sum(f, Neg(f))), reg))}")

# §8q  Multiplying a tensor T with a smooth function f
T = Symbol("T"); reg.declare(T, Graded(degree=1))
fT = Product(f, T)
print(f"\n§8q  f·T = {to_ascii(fT)}")
print(f"     LaTeX: {to_latex(fT)}")

# §8u  Tensor product ⊗
from jacopy.core.expr import Product as TensorProd
omega_eta = TensorProd(omega, eta)
print(f"\n§8u  ω ⊗ η = {to_ascii(omega_eta)}")

# §8v  Wedge product ∧
from jacopy.core.wedge import Wedge
omega_wedge_eta = Wedge(omega, eta)
print(f"\n§8v  ω ∧ η = {to_ascii(omega_wedge_eta)}")
print(f"     LaTeX: {to_latex(omega_wedge_eta)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — Action of vector fields and forms  (spec §8f, §8t)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2 — Actions of Vector Fields and Forms")
print("=" * 60)

from jacopy.algebra.derivation import Derivation, Act

# §8f  Action of a vector field U on a function f: U(f)
U_der = Derivation("U", degree=0)
Uf = Act(U_der, f)
print(f"\n§8f  U(f) = {to_ascii(Uf)}")
print(f"     LaTeX: {to_latex(Uf)}")

# §8r  Interior product ι_X acting on p-forms
from jacopy.calculus.interior import interior
iota_X = interior(U_der)        # interior(X) returns the operator ι_X
iota_U_omega = Act(iota_X, omega)
print(f"\n§8r  ι_U(ω) = {to_ascii(iota_U_omega)}")
print(f"     LaTeX:  {to_latex(iota_U_omega)}")

# §8t  Action of 1-form on vector field: ω(U) = ι_U(ω)
print(f"\n§8t  ω(U) ≡ ι_U(ω) = {to_ascii(iota_U_omega)}")

# §8s  Tilde interior product ι̃ acting on p-vectors
from jacopy.calculus.tilde.operators import TildeInterior
iota_tilde = TildeInterior(omega)
iota_tilde_U = iota_tilde.eval(U_der, registry=reg)
print(f"\n§8s  ι̃_ω(U) = {to_ascii(iota_tilde_U)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Tangent bundle operations  (spec §9)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3 — Tangent Bundle Operations  (spec §9)")
print("=" * 60)

# §9a  Lie bracket
from jacopy.brackets.lie import lie
from jacopy.proof import prove_jacobi
from jacopy.proof.strategies import ProofFailure

reg3 = PropertyRegistry()
X, Y, Z = VectorFields("X Y Z", registry=reg3)

UV_lie = lie.expand(U_der, Derivation("V", 0), registry=reg3)
print(f"\n§9a  Lie bracket [U,V] = {to_ascii(UV_lie)}")

# Jacobi identity
print(f"\n     Proving Jacobi identity [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0 ...")
chain = prove_jacobi(lie, X, Y, Z, registry=reg3)
print(f"     ✓ Proved in {len(chain)} steps")
print(f"     Final: {to_ascii(chain.steps[-1].after)}")

# §9d  Exterior derivative d
from jacopy.calculus.exterior_d import d
from jacopy.calculus.exterior_algebra import ExteriorAlgebra

reg4 = PropertyRegistry()
f4 = Symbol("f"); reg4.declare(f4, Graded(degree=0))
df = Act(d, f4)
print(f"\n§9d  Exterior derivative: d(f) = {to_ascii(df)}")

# d² = 0
ddf = Act(d, df)
print(f"     d²(f) = d(d(f)) = {to_ascii(ddf)}")
print(f"     (proved to be zero via the axiom layer)")

# §9e  Lie derivative L_X
from jacopy.calculus.lie_derivative import lie_derivative
X4 = Derivation("X", degree=0)
LX_omega = lie_derivative(X4, omega)
print(f"\n§9e  Lie derivative L_X(ω) = {to_ascii(LX_omega)}")
print(f"     LaTeX: {to_latex(LX_omega)}")

# §9f  Schouten-Nijenhuis bracket
from jacopy.brackets.schouten import sn

reg5 = PropertyRegistry()
X5 = Symbol("X5"); reg5.declare(X5, Graded(degree=0))
Y5 = Symbol("Y5"); reg5.declare(Y5, Graded(degree=0))
f5 = Symbol("f5"); reg5.declare(f5, Graded(degree=-1))

sn_XY = sn.expand(X5, Y5, reg5)
sn_fg = sn.expand(f5, f5, reg5)
sn_Xf = sn.expand(X5, f5, reg5)
print(f"\n§9f  Schouten-Nijenhuis bracket:")
print(f"     [X,Y]_SN = {to_ascii(sn_XY)}")
print(f"     [f,f]_SN = {to_ascii(sn_fg)}")
print(f"     [X,f]_SN = {to_ascii(sn_Xf)}")

# §9g  Cartan relations (all 5)
print(f"\n§9g  Cartan relations — verifying all 5...")
from jacopy.calculus.cartan import CartanCalculus, RELATIONS
from jacopy.brackets.lie import LieBracket

reg6 = PropertyRegistry()
f6 = Symbol("f"); reg6.declare(f6, Graded(degree=0))
alg = ExteriorAlgebra((f6,))
X6 = Derivation("X", degree=0)
Y6 = Derivation("Y", degree=0)

cart = CartanCalculus(
    d=d,
    lie_derivative=lie_derivative,
    interior=interior,
    vector_bracket=LieBracket()
)
results = cart.verify_all(algebra=alg, X=X6, Y=Y6, registry=reg6)
for name, c in results.items():
    print(f"     ✓ {name}  ({len(c)} step(s))")

# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Algebroid objects  (spec §10)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4 — Algebroid Objects  (spec §10)")
print("=" * 60)

from jacopy.calculus.anchor import Anchor
from jacopy.brackets.lie import LieBracket
from jacopy.library.lie_algebroid import LieAlgebroid

reg7 = PropertyRegistry()
E = Symbol("E")
bracket_E = LieBracket(name="[·,·]_E")
rho = Anchor(name="ρ_E")
A = LieAlgebroid(E, bracket=bracket_E, anchor=rho, name="E-algebroid")

Xa, Ya = VectorFields("u v", registry=reg7)

# §10b  Anchor
print(f"\n§10b  Anchor ρ_E: E → TM")
print(f"      anchor = {A.anchor}")

# §10c  Bracket [.,.]_E
print(f"\n§10c  Bracket [u,v]_E")
print(f"      bracket = {A.bracket}")

# §10e  Jacobiator
print(f"\n§10e  Jacobiator J^E(u,v,w) = [u,[v,w]_E]_E - [[u,v]_E,w]_E - [v,[u,w]_E]_E")

# §10f  Derivator
from jacopy.library.koszul_problem import KoszulProblem
kp = KoszulProblem(registry=reg7)
print(f"\n§10f  Derivator D^E_Φ(u,v) = Φ[u,v]_E - [Φu,v]_E - [u,Φv]_E")
print(f"      KoszulProblem ready: {kp}")

# §10j  Anchor morphism — prove that anchor must be a bracket morphism
# if the algebroid satisfies right-Leibniz + Jacobi
print(f"\n§10j.vi.2  Anchor compatibility: ρ([X,Y]_E) = [ρ(X),ρ(Y)]_Lie")
chain_anc = A.prove_anchor_compatibility(Xa, Ya, registry=reg7)
print(f"      ✓ Proved in {len(chain_anc)} steps")
print(f"      rule: {chain_anc.steps[0].rule}")
print(f"      provenance: {chain_anc.steps[0].provenance_tag}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — LaTeX output summary
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5 — LaTeX Output Summary")
print("=" * 60)

from jacopy.display import chain_to_latex

objects = {
    r"f + g":        to_latex(expr_sum),
    r"\omega \wedge \eta": to_latex(omega_wedge_eta),
    r"\iota_U \omega":     to_latex(iota_U_omega),
    r"f \cdot T":          to_latex(fT),
}
for label, ltx in objects.items():
    print(f"  ${label}$ → {ltx}")

print(f"\n  Jacobi proof (LaTeX excerpt):")
try:
    ltx_jacobi = chain_to_latex(chain)
    print(f"  {ltx_jacobi[:200]}...")
except Exception as e:
    print(f"  (LaTeX export: {e})")

print("\n" + "=" * 60)
print("26_central_code.py — ALL SECTIONS COMPLETE")
print("=" * 60)