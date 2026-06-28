"""
29_symplectic_poisson.py
========================
Symplectic, Poisson and Nambu-Poisson Geometry
Covers spec §12:
  §12a   Symplectic ⊂ Poisson relation
  §12c   Poisson bivector (pi)
  §12g   Koszul bracket [.,.]_Kos on T*M
  §12h   All algebroid properties of the Koszul bracket
  §12j   Tilde exterior derivative d̃ (Lichnerowicz)
  §12k   Tilde Lie derivative L̃
  §12l   All Cartan relations (tilde side)
  §12p   Roytenberg bracket

Usage:
    python 29_symplectic_poisson.py
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
from jacopy.display import to_ascii, to_latex, chain_to_latex
from jacopy.proof.strategies import ProofFailure
from jacopy.library.declarations import Bivector, Functions
from jacopy.library.poisson import PoissonBracket
from jacopy.library.symplectic import SymplecticManifold
from jacopy.library import theorem_book
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.musical import Sharp
from jacopy.calculus.tilde import (
    tilde_interior, tilde_d, tilde_lie,
    tilde_intrinsic_engine, prove_tilde_cartan_relation,
)

# ── shared registry ───────────────────────────────────────────
reg = PropertyRegistry()
pi  = Bivector("π", registry=reg)
reg.declare(pi, Poisson())

(omega,) = Forms("ω", degree=2, registry=reg)
f, g, h  = Functions("f g h", degree=-1, registry=reg)
alpha, beta, gamma = Forms("α β γ", degree=1, registry=reg)

V = Symbol("V"); reg.declare(V, Graded(degree=1))
W = Symbol("W"); reg.declare(W, Graded(degree=2))
eta = Symbol("η"); reg.declare(eta, Graded(degree=1))
mu  = Symbol("μ"); reg.declare(mu,  Graded(degree=1))

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — Symplectic manifold & musical maps  (spec §12a)
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SECTION 1 — Symplectic Manifold  (spec §12a)")
print("=" * 60)

M = SymplecticManifold(omega, bivector=pi, name="(M, ω, π)")
print(f"\n§12a  Symplectic ⊂ Poisson:")
print(f"      A symplectic manifold is a Poisson manifold with")
print(f"      non-degenerate π satisfying ω^♭ ∘ π^♯ = id")
print(f"\n      flat  (ω^♭) : {M.flat}")
print(f"      sharp (π^♯) : {M.sharp}")
print(f"      compatibility: {M.compatibility}")

# Hamiltonian equivalence proof
print(f"\n  Proving Hamiltonian equivalence ι_{{X_f}} ω + df = 0 ...")
chain_ham = M.prove_hamiltonian_equivalence(f, registry=reg)
print(f"  ✓ Proved in {len(chain_ham)} steps")
for i, step in enumerate(chain_ham.steps, 1):
    print(f"    {i}. {step.rule}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — Poisson bracket: three views  (spec §12c)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2 — Poisson Bracket: Three Views  (spec §12c)")
print("=" * 60)

poisson = PoissonBracket.from_bivector(pi)

# View 1: derived bracket
derived = poisson.expand(f, g, reg)
print(f"\n  View 1 — Derived:  {{f,g}}_π = {to_ascii(derived)}")

# View 2: Hamiltonian vector field
ham = poisson.via_hamiltonian(f, g)
Xf  = poisson.hamiltonian_vf(f)
print(f"\n  View 2 — Hamiltonian:  {{f,g}}_π = X_f(g) = {to_ascii(ham)}")
print(f"           Hamiltonian vf X_f = {to_ascii(Xf)}")

# View 3: Koszul three-term formula
koszul_exp = poisson.koszul_expand(alpha, beta, reg)
print(f"\n  View 3 — Koszul (on forms):")
print(f"    {{α,β}}_π = L_{{π♯(α)}}β − L_{{π♯(β)}}α − d⟨π♯(α),β⟩")
print(f"    = {to_ascii(koszul_exp)}")
print(f"    LaTeX: {to_latex(koszul_exp)}")

# Koszul equivalence (1-step reflexive)
chain_keq = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)
print(f"\n  Koszul equivalence proof: {len(chain_keq)} step,  rule={chain_keq.steps[0].rule}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Jacobi identity & [π,π]_SN=0  (spec §12c)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3 — Jacobi Identity & [π,π]_SN = 0  (spec §12c)")
print("=" * 60)

obs = poisson.jacobi_obstruction(reg)
cond = poisson.jacobi_condition(reg)
print(f"\n  Jacobi obstruction: {to_ascii(obs)}")
print(f"  LaTeX: {to_latex(obs)}")
print(f"  Condition name: {cond.name}")

# Jacobi reduction chain
chain_jac = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
print(f"\n  Jacobi reduction: {len(chain_jac)} step")
print(f"  rule: {chain_jac.steps[0].rule}")
print(f"  after: {to_ascii(chain_jac.steps[0].after)}")

# Koszul Jacobi reduction (form level)
chain_kjac = poisson.prove_koszul_jacobi_reduction(alpha, beta, gamma, registry=reg)
print(f"\n  Koszul Jacobi reduction (forms): {len(chain_kjac)} step")
print(f"  rule: {chain_kjac.steps[0].rule}")

# theorem_book
thm = theorem_book.get("poisson_jacobi")
print(f"\n  theorem_book['poisson_jacobi']:")
print(f"  statement  : {thm.statement}")
print(f"  from_axioms: {thm.from_axioms}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Koszul bracket on T*M  (spec §12g, §12h)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4 — Koszul Bracket on T*M  (spec §12g, §12h)")
print("=" * 60)

sharp  = Sharp(pi)
koszul = KoszulBracket(sharp)

koz_ab = koszul.expand(alpha, beta)
print(f"\n§12g  Koszul bracket [α,β]_K = {to_ascii(koz_ab)}")
print(f"      LaTeX: {to_latex(koz_ab)}")

print(f"\n§12h  Algebroid properties of the Koszul bracket:")
print(f"      is_graded_antisymmetric : {koszul.is_graded_antisymmetric}")
print(f"      satisfies_leibniz       : {koszul.satisfies_leibniz}")
print(f"      satisfies_graded_jacobi : {koszul.satisfies_graded_jacobi}")
print(f"      (Koszul bracket is a Lie algebroid on T*M)")

# Jacobi for Koszul bracket (from DerivedBracket path)
from jacopy.brackets.derived import DerivedBracket
from jacopy.proof import prove_jacobi

kd = DerivedBracket(sn, pi, degree_Q=1, acting_on=sharp)
print(f"\n  Koszul as derived bracket (DerivedBracket):")
print(f"  jacobi_obstruction_raw: {kd.jacobi_obstruction_raw()}")
print(f"  condition.holds (Poisson declared): {kd.jacobi_condition(reg).holds(reg)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Tilde calculus  (spec §12j, §12k)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5 — Tilde Calculus  (spec §12j, §12k)")
print("=" * 60)

i_til = tilde_interior(eta)
d_til = tilde_d(pi)
L_til = tilde_lie(eta, pi)

print(f"\n§12j  Tilde exterior derivative:")
print(f"      d̃ (Lichnerowicz): d̃V = [π, V]_SN")
print(f"      atom: {d_til},  degree: {d_til._degree}")

expr_dV = Act(d_til, V)
print(f"      d̃V = {to_ascii(expr_dV)}")
print(f"      LaTeX: {to_latex(expr_dV)}")

print(f"\n§12k  Tilde Lie derivative:")
print(f"      L̃_ω V = d̃ ι̃_ω V + ι̃_ω d̃ V  (Cartan magic)")
print(f"      atom: {L_til},  degree: {L_til._degree}")

expr_LV = Act(L_til, V)
print(f"      L̃_ω V = {to_ascii(expr_LV)}")

print(f"\n  Tilde interior product:")
print(f"      ι̃_ω V := ι_V ω  (notation swap)")
print(f"      atom: {i_til},  degree: {i_til._degree}")
expr_iV = Act(i_til, V)
print(f"      ι̃_η V = {to_ascii(expr_iV)}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Tilde Cartan relations  (spec §12l)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 6 — Tilde Cartan Relations  (spec §12l)")
print("=" * 60)

eng = tilde_intrinsic_engine(pi, koszul, sharp=sharp, registry=reg)

# Magic formula: L̃_η = d̃ ι̃_η + ι̃_η d̃
lhs_magic = Act(L_til, V)
rhs_magic = Sum(Act(d_til, Act(i_til, V)), Act(i_til, Act(d_til, V)))
chain_magic = prove_tilde_cartan_relation(
    lhs_magic, rhs_magic, etas=(eta,), engine=eng, registry=reg,
)
print(f"\n  Magic formula L̃_η = d̃ ι̃_η + ι̃_η d̃:")
print(f"  ✓ Proved in {len(chain_magic)} steps")

# Anti-commute: ι̃_η ι̃_μ + ι̃_μ ι̃_η = 0
i_mu = tilde_interior(mu)
lhs_anti = Sum(Act(i_til, Act(i_mu, W)), Act(i_mu, Act(i_til, W)))
chain_anti = prove_tilde_cartan_relation(
    lhs_anti, Integer(0), etas=(eta,), engine=eng, registry=reg,
)
print(f"\n  Anti-commute ι̃_η ι̃_μ + ι̃_μ ι̃_η = 0:")
print(f"  ✓ Proved in {len(chain_anti)} steps")

# d̃² = 0  (Poisson flag declared)
from jacopy.calculus.tilde import TildeDSquaredPoissonDefinition
from jacopy.proof.expansion import ExpansionEngine

engine_dsq = ExpansionEngine([TildeDSquaredPoissonDefinition(pi, registry=reg)])
expr_dsq = Act(d_til, Act(d_til, V))
out_dsq, steps_dsq = engine_dsq.expand(expr_dsq)
print(f"\n  d̃² = 0  (Jacobi identity in dual form):")
print(f"  d̃(d̃V) → {to_ascii(out_dsq)}")
print(f"  rule: {steps_dsq[0].rule}")

# ═════════════════════════════════════════════════════════════════════
# SECTION 7 — Roytenberg bracket  (spec §12p)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 7 — Roytenberg Bracket  (spec §12p)")
print("=" * 60)

print(f"\n§12p  Roytenberg bracket on TM ⊕ T*M:")
print(f"      [a, b]_Roy := [a, b]_Dorf + correction from π")
print(f"      (generalises Courant algebroid to Poisson setting)")

try:
    from jacopy.brackets.courant_lwx import RoytenbergBracket
    reg_roy = PropertyRegistry()
    pi_roy = Bivector("π", registry=reg_roy)
    reg_roy.declare(pi_roy, Poisson())
    X_roy, Y_roy = VectorFields("X Y", registry=reg_roy)
    alpha_roy, beta_roy = Forms("α β", degree=1, registry=reg_roy)
    from jacopy.brackets.dorfman import SectionPair
    a_roy = SectionPair(X_roy, alpha_roy)
    b_roy = SectionPair(Y_roy, beta_roy)
    roy = RoytenbergBracket(pi_roy)
    result_roy = roy.expand(a_roy, b_roy, registry=reg_roy)
    print(f"\n  [a,b]_Roy  vector: {to_ascii(result_roy.vector)}")
    print(f"            form  : {to_ascii(result_roy.form)}")
    print(f"            LaTeX : {to_latex(result_roy.form)}")
except ImportError:
    # fallback: describe via Dorfman + tilde
    print(f"\n  RoytenbergBracket not directly importable;")
    print(f"  Roytenberg = Dorfman bracket twisted by Poisson bivector π.")
    print(f"  When π=0: reduces to standard Courant/Dorfman bracket.")
    print(f"  General form: [a,b]_Roy = [a,b]_Dorf + (0, ι_X ι_Y dπ)")

# ═════════════════════════════════════════════════════════════════════
# SECTION 8 — LaTeX summary
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 8 — LaTeX Summary")
print("=" * 60)

exprs = {
    r"\{f,g\}_\pi \text{ (derived)}":   to_latex(derived),
    r"\{f,g\}_\pi \text{ (Hamiltonian)}": to_latex(ham),
    r"\{\alpha,\beta\}_\pi \text{ (Koszul)}": to_latex(koszul_exp),
    r"[\alpha,\beta]_K":                to_latex(koz_ab),
    r"\tilde{d} V":                     to_latex(expr_dV),
    r"\tilde{\iota}_\eta V":            to_latex(expr_iV),
    r"[\pi,\pi]_{SN}":                  to_latex(obs),
}
for label, ltx in exprs.items():
    print(f"  ${label}$")
    print(f"    → {ltx}")

print("\n" + "=" * 60)
print("29_symplectic_poisson.py — ALL SECTIONS COMPLETE")
print("=" * 60)
