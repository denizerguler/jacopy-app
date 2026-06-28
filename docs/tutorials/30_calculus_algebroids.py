"""
30_calculus_algebroids.py
=========================
Calculus on Algebroids
Covers spec §13a-g
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from jacopy import VectorFields, Forms
from jacopy.core.expr import Symbol, Sum, Neg, Integer
from jacopy.core.properties import Graded, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.algebra.derivation import Derivation, Act
from jacopy.display import to_ascii, to_latex
from jacopy.brackets.lie import LieBracket
from jacopy.brackets.schouten import sn as default_sn
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.cartan import CartanCalculus, RELATIONS
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.cartan_remainder import K
from jacopy.calculus.tilde import (
    tilde_interior, tilde_d, tilde_lie, K_tilde,
    tilde_intrinsic_engine, prove_tilde_cartan_relation,
    TildeDSquaredPoissonDefinition,
)
from jacopy.calculus.derivator import derivator
from jacopy.library.lie_algebroid import LieAlgebroid
from jacopy.library.koszul_problem import KoszulProblem
from jacopy.library.declarations import Bivector
from jacopy.calculus.musical import Sharp
from jacopy.proof.expansion import ExpansionEngine

# ── SECTION 1: Algebroid Cartan Calculus ─────────────────────
print("=" * 60)
print("SECTION 1 — Algebroid Cartan Calculus  (spec §13a, §13e)")
print("=" * 60)

reg_alg = PropertyRegistry()
E = Symbol("E")
bracket_E = LieBracket(name="[·,·]_E")
rho = Anchor(name="rho_E")
A = LieAlgebroid(E, bracket=bracket_E, anchor=rho, name="E-algebroid")
X, Y = VectorFields("X Y", registry=reg_alg)

cart = A.cartan
print(f"\n§13a  Algebroid operators on E:")
print(f"      d_E     = {A.d}")
print(f"      L_{{E,X}} = {cart.lie_derivative(X)}")
print(f"      i_{{E,X}} = {cart.interior(X)}")

chain_anc = A.prove_anchor_compatibility(X, Y, registry=reg_alg)
print(f"\n§13e  Anchor compatibility rho([X,Y]_E) = [rho(X),rho(Y)]_TM:")
print(f"      Proved in {len(chain_anc)} step, rule={chain_anc.steps[0].rule}")

reg6 = PropertyRegistry()
f6 = Symbol("f"); reg6.declare(f6, Graded(degree=0))
alg = ExteriorAlgebra((f6,))
X6 = Derivation("X", degree=0)
Y6 = Derivation("Y", degree=0)
cart_tm = CartanCalculus(
    d=default_d, lie_derivative=lie_derivative,
    interior=interior, vector_bracket=LieBracket()
)
results = cart_tm.verify_all(algebra=alg, X=X6, Y=Y6, registry=reg6)
print(f"\n§13e  Five Cartan relations on TM:")
for name, chain in results.items():
    print(f"      OK {name}  ({len(chain)} step(s))")

# ── SECTION 2: K and K̃ operators ──────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — K and K̃ Operators  (spec §13b)")
print("=" * 60)

reg2 = PropertyRegistry()
pi2   = Symbol("pi");  reg2.declare(pi2,   Graded(degree=1)); reg2.declare(pi2, Poisson())
omega2 = Symbol("omega"); reg2.declare(omega2, Graded(degree=1))
eta2   = Symbol("eta");   reg2.declare(eta2,   Graded(degree=1))
U2     = Symbol("U");     reg2.declare(U2,     Graded(degree=1))

K_U2 = K(U2)
print(f"\n§13b  K_V (Cartan remainder, form side):")
print(f"      K_V := -L_V + d ∘ i_V")
print(f"      K_U = {K_U2}, degree={K_U2._degree}")
expr_K = Act(K_U2, omega2)
print(f"      K_U(omega) = {to_ascii(expr_K)}")
print(f"      LaTeX: {to_latex(expr_K)}")

K_til_eta = K_tilde(eta2, pi2)
print(f"\n§13b  K~_eta (tilde Cartan remainder, multivector side):")
print(f"      K~_eta := -L~_eta + d~ ∘ i~_eta")
print(f"      K~_eta = {K_til_eta}, degree={K_til_eta._degree}")
print(f"      form={K_til_eta.form}, bivector={K_til_eta.bivector}")
expr_Ktil = Act(K_til_eta, U2)
print(f"      K~_eta(U) = {to_ascii(expr_Ktil)}")
print(f"      LaTeX: {to_latex(expr_Ktil)}")

# ── SECTION 3: Tilde calculus ─────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — Tilde Calculus on Poisson Manifold  (spec §13f)")
print("=" * 60)

reg3 = PropertyRegistry()
pi3   = Bivector("pi",  registry=reg3); reg3.declare(pi3, Poisson())
eta3  = Symbol("eta");  reg3.declare(eta3,  Graded(degree=1))
mu3   = Symbol("mu");   reg3.declare(mu3,   Graded(degree=1))
V3    = Symbol("V");    reg3.declare(V3,    Graded(degree=1))
W3    = Symbol("W");    reg3.declare(W3,    Graded(degree=2))

i_til3 = tilde_interior(eta3)
d_til3 = tilde_d(pi3)
L_til3 = tilde_lie(eta3, pi3)
i_mu3  = tilde_interior(mu3)

sharp3  = Sharp(pi3)
koszul3 = KoszulBracket(sharp3)
eng3    = tilde_intrinsic_engine(pi3, koszul3, sharp=sharp3, registry=reg3)

lhs_magic = Act(L_til3, V3)
rhs_magic = Sum(Act(d_til3, Act(i_til3, V3)), Act(i_til3, Act(d_til3, V3)))
chain_magic = prove_tilde_cartan_relation(
    lhs_magic, rhs_magic, etas=(eta3,), engine=eng3, registry=reg3,
)
print(f"\n§13f  Tilde magic formula L~_eta = d~ i~_eta + i~_eta d~:")
print(f"      Proved in {len(chain_magic)} steps")

lhs_anti = Sum(Act(i_til3, Act(i_mu3, W3)), Act(i_mu3, Act(i_til3, W3)))
chain_anti = prove_tilde_cartan_relation(
    lhs_anti, Integer(0), etas=(eta3,), engine=eng3, registry=reg3,
)
print(f"\n      Tilde anti-commute i~_eta i~_mu + i~_mu i~_eta = 0:")
print(f"      Proved in {len(chain_anti)} steps")

engine_dsq = ExpansionEngine([TildeDSquaredPoissonDefinition(pi3, registry=reg3)])
expr_dsq = Act(d_til3, Act(d_til3, V3))
out_dsq, steps_dsq = engine_dsq.expand(expr_dsq)
print(f"\n      d~^2 = 0 (Jacobi in dual form):")
print(f"      d~(d~V) -> {to_ascii(out_dsq)}, rule={steps_dsq[0].rule}")

# ── SECTION 4: Derivator identities ──────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — Derivator Identities 3.1.5  (spec §13c, §13g)")
print("=" * 60)

print(f"\n  Derivator D^E_phi(u,v) := phi[u,v]_E - [phi u,v]_E - (-1)^(d|u|) [u,phi v]_E")
print(f"  phi is a derivation of [.,.]_E iff D^E_phi = 0")

reg4 = PropertyRegistry()
pi4   = Symbol("pi")
omega4, eta4, mu4 = Symbol("omega"), Symbol("eta"), Symbol("mu")
U4, V4, W4 = Symbol("U"), Symbol("V"), Symbol("W")
for sym in (pi4, omega4, eta4, mu4, U4, V4, W4):
    reg4.declare(sym, Graded(degree=1))

Y4 = Derivation("Y", 0)
xi4 = Symbol("xi"); reg4.declare(xi4, Graded(degree=1))

prob = KoszulProblem(
    pi4, (omega4, eta4, mu4),
    registry=reg4,
    multivectors=((U4, 1), (V4, 1), (W4, 1)),
)
prob.assume_poisson()
K_b = prob.koszul_bracket

print(f"\n  KoszulProblem engine rules:")
print(f"    form engine        : {len(prob.derivator_form_engine().definitions)}")
print(f"    multivector engine : {len(prob.derivator_multivector_engine().definitions)}")

# Identity (1) form side
print(f"\n  Identity (1) form side:")
print(f"  D^T*M_{{L_U}}(eta,mu) = L_{{K~_eta U}} mu + K_{{K~_mu U}} eta")
lhs1 = derivator(lie_derivative(U4), K_b, eta4, mu4)
K_tilde_eta_U = Act(K_tilde(eta4, pi4), U4)
K_tilde_mu_U  = Act(K_tilde(mu4,  pi4), U4)
rhs1 = Sum(
    Act(lie_derivative(K_tilde_eta_U), mu4),
    Act(K(K_tilde_mu_U), eta4),
)
chain1 = prob.prove_derivator(lhs1, rhs1, eval_args=(Y4,), side="form")
print(f"  Proved in {len(chain1)} steps")

# Identity (1') multivector side
print(f"\n  Identity (1') multivector side:")
print(f"  D~^SN_{{L~_eta}}(U,V) = L~_{{K_U eta}} V + K~_{{K_V eta}} U")
lhs1p = derivator(tilde_lie(eta4, pi4), default_sn, U4, V4)
K_U4_eta = Act(K(U4), eta4)
K_V4_eta = Act(K(V4), eta4)
rhs1p = Sum(
    Act(tilde_lie(K_U4_eta, pi4), V4),
    Act(K_tilde(K_V4_eta, pi4), U4),
)
chain1p = prob.prove_derivator(lhs1p, rhs1p, eval_args=(xi4,), side="multivector")
print(f"  Proved in {len(chain1p)} steps")

print(f"\n  Summary:")
print(f"  (1)  form         {len(chain1)} steps")
print(f"  (1') multivector  {len(chain1p)} steps")
print(f"  (2), (2'), (3), (3') require BracketApply API — see tutorial 18")

# ── SECTION 5: Compatibility conditions ───────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — Compatibility Conditions  (spec §13d, §13g)")
print("=" * 60)

print(f"\n§13d  Key compatibility relations:")
print(f"  1. [d, i~_omega] = L~_omega")
print(f"  2. [d~, i_X] = L_X")
print(f"  3. d d~ + d~ d = 0  (when [pi,pi]_SN = 0)")

from jacopy.library.declarations import Functions
from jacopy.library.poisson import PoissonBracket

reg5 = PropertyRegistry()
pi5 = Bivector("pi", registry=reg5); reg5.declare(pi5, Poisson())
f5, g5, h5 = Functions("f g h", degree=-1, registry=reg5)
alpha5, beta5, gamma5 = Forms("alpha beta gamma", degree=1, registry=reg5)
pb5 = PoissonBracket.from_bivector(pi5)

chain_func = pb5.prove_jacobi_reduction(f5, g5, h5, registry=reg5)
chain_form = pb5.prove_koszul_jacobi_reduction(alpha5, beta5, gamma5, registry=reg5)

print(f"\n  Same obstruction [pi,pi]_SN=0 from both sides:")
print(f"  Function Jacobi: rule={chain_func.steps[0].rule}")
print(f"  Form Jacobi    : rule={chain_form.steps[0].rule}")
same = str(chain_func.steps[0].after) == str(chain_form.steps[0].after)
print(f"  Same obstruction? {same}")

# ── SECTION 6: LaTeX summary ───────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — LaTeX Summary")
print("=" * 60)

exprs = {
    "K_U(omega)":      to_latex(expr_K),
    "K~_eta(U)":       to_latex(expr_Ktil),
    "d~(d~V)":         to_latex(expr_dsq),
    "i~_eta(V)":       to_latex(Act(i_til3, V3)),
}
for label, ltx in exprs.items():
    print(f"  {label}  ->  {ltx}")

print("\n" + "=" * 60)
print("30_calculus_algebroids.py — ALL SECTIONS COMPLETE")
print("=" * 60)
