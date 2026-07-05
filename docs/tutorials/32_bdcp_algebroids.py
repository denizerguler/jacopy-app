"""
32_bdcp_algebroids.py
======================
Bicocycle Double Cross Product (BDCP) Lie Algebroids
(spec: Ateşli, Esen & Sütlü — "On Product Lie Algebroids, and
 Collective Motion", arXiv:2503.12059, 2025)

This module builds a GENERAL framework for the six-structure-map
hierarchy of the paper:

    unified product  ⊂  double cross product  ⊂  BDCP
    2-cocycle ext.    ⊂  semi-direct product  ⊂  BDCP

using ONLY Jacopy primitives that are already verified elsewhere in
this project (CustomBracket, prove_jacobi, PropertyRegistry, Symbol,
Sum, Neg, Integer). No unverified jacopy-internal API is invented.

WHAT IS ACTUALLY NEW HERE (built from scratch, not from Jacopy core):
  - BDCPSection    : a lightweight pair type (a,b) ∈ A × B, playing
                     the role SectionPair plays for Dorfman/Courant,
                     but written independently since we don't know
                     whether jacopy's SectionPair generalizes to
                     arbitrary A,B (only TM ⊕ T*M is verified).
  - BDCPBracket    : combines six user-supplied component functions
                     (phi, zeta, psi, theta, rho, sigma) into ONE
                     bracket function usable by CustomBracket, and
                     classifies which corner of the hierarchy it
                     sits in (exactly the checkboxes in Tab VII of
                     app.py, but computed once from data instead of
                     re-derived by hand in each button callback).

WHAT IS STILL MISSING (honest — not faked here):
  - Vector-bundle-valued structure maps (this module only handles
    the finite-dimensional / scalar-multiplicity case, same as the
    two worked examples in Tab VII).
  - A contactization / z-extension rule for the dissipative
    Hamilton's equations of §3 of the paper (would live alongside
    the K/K̃ Cartan-remainder operators in 15_intrinsic_engine /
    17_tilde_calculus if it is ever added).

Usage:
    python 32_bdcp_algebroids.py
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from jacopy.core.expr import Symbol, Sum, Neg, Integer, Expr
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ProofFailure
from jacopy.proof import prove_jacobi
from jacopy.brackets.custom import CustomBracket
from jacopy.display import to_ascii, chain_to_ascii

print("=" * 60)
print("BICOCYCLE DOUBLE CROSS PRODUCT (BDCP) LIE ALGEBROIDS")
print("(Ateşli, Esen & Sütlü, arXiv:2503.12059)")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — BDCPSection: a pair type for A × B
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 1 — BDCPSection (pair type for A × B)")
print("=" * 60)

class BDCPSection:
    """
    A section of A × B, i.e. a pair (a, b) with a ∈ A, b ∈ B.

    This is the 'direct-sum expression type' identified as missing
    in the app.py roadmap. It plays the same structural role that
    SectionPair plays for TM ⊕ T*M in the Dorfman/Courant tutorials,
    but is written independently here since the paper's A, B are
    arbitrary vector bundles, not necessarily TM and T*M.

    Component-wise arithmetic is provided so that a BDCPBracket can
    return a BDCPSection directly, mirroring the six-term bracket
    formula in the paper:

        [(a1,b1),(a2,b2)] = ( φ(a1,a2) - ρ(b2,a1) + ρ(b1,a2) + ψ(b1,b2),
                              θ(b1,b2) - σ(a2,b1) + σ(a1,b2) + ζ(a1,a2) )
    """

    __slots__ = ("a", "b")

    def __init__(self, a: Expr, b: Expr):
        self.a = a
        self.b = b

    def __repr__(self):
        return f"({to_ascii(self.a)}, {to_ascii(self.b)})"

    def __add__(self, other: "BDCPSection") -> "BDCPSection":
        return BDCPSection(Sum(self.a, other.a), Sum(self.b, other.b))

    def __neg__(self) -> "BDCPSection":
        return BDCPSection(Neg(self.a), Neg(self.b))

    def __sub__(self, other: "BDCPSection") -> "BDCPSection":
        return self + (-other)

    def a_part(self) -> Expr: return self.a
    def b_part(self) -> Expr: return self.b


# Demo
print("""
  BDCPSection((a,b)):
""")
a1 = Symbol("a1"); b1 = Symbol("b1")
a2 = Symbol("a2"); b2 = Symbol("b2")
sec1 = BDCPSection(a1, b1)
sec2 = BDCPSection(a2, b2)
print(f"  sec1 = {sec1}")
print(f"  sec2 = {sec2}")
print(f"  sec1 + sec2 = {sec1 + sec2}")
print(f"  sec1 - sec2 = {sec1 - sec2}")


# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — BDCPBracket: combine six structure maps into one bracket
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2 — BDCPBracket (six structure maps → one bracket)")
print("=" * 60)

# Default trivial (zero) structure map
def _zero_map(x, y, registry=None):
    return Integer(0)

class BDCPBracket:
    """
    Combines six structure maps

        phi   : A × A → A     (bracket internal to A)
        zeta  : A × A → B     (twisted 2-cocycle, A-side)
        psi   : B × B → A     (twisted 2-cocycle, B-side)
        theta : B × B → B     (bracket internal to B)
        rho   : B × A → A     (weak/genuine action of B on A)
        sigma : A × B → B     (weak/genuine action of A on B)

    into the single BDCP bracket on BDCPSection pairs:

        [(a1,b1),(a2,b2)]_ζ⋈ψ =
            ( phi(a1,a2) - rho(b2,a1) + rho(b1,a2) + psi(b1,b2),
              theta(b1,b2) - sigma(a2,b1) + sigma(a1,b2) + zeta(a1,a2) )

    exactly matching the formula in the paper (their eq. for the
    BDCP bracket, restated in app.py's Tab VII intro).

    Each map defaults to the zero map, so constructing a BDCPBracket
    with only a subset of maps supplied automatically produces the
    correct specialization (unified product, double cross product,
    semi-direct product, 2-cocycle extension, or plain direct sum).
    """

    def __init__(self, phi=None, zeta=None, psi=None,
                 theta=None, rho=None, sigma=None, name="[·,·]_BDCP"):
        self.phi   = phi   or _zero_map
        self.zeta  = zeta  or _zero_map
        self.psi   = psi   or _zero_map
        self.theta = theta or _zero_map
        self.rho   = rho   or _zero_map
        self.sigma = sigma or _zero_map
        self.name  = name

    # ── classification, mirrors the Hierarchy Explorer in app.py ──
    def _is_trivial(self, fn) -> bool:
        return fn is _zero_map

    @property
    def zeta_trivial(self):  return self._is_trivial(self.zeta)
    @property
    def psi_trivial(self):   return self._is_trivial(self.psi)
    @property
    def rho_trivial(self):   return self._is_trivial(self.rho)
    @property
    def sigma_trivial(self): return self._is_trivial(self.sigma)
    @property
    def phi_trivial(self):   return self._is_trivial(self.phi)
    @property
    def theta_trivial(self): return self._is_trivial(self.theta)

    def classify(self) -> str:
        """
        Names the specialization of the BDCP hierarchy this bracket
        realizes, following the Introduction of the paper (and the
        exact same logic as the Hierarchy Explorer tab in app.py).
        """
        z, p, r, s = self.zeta_trivial, self.psi_trivial, self.rho_trivial, self.sigma_trivial
        if z and p and not r and not s:
            return "Double Cross Product (matched pair)  A ⋈ B"
        if z and p and s and not r:
            return "Semi-direct Product  A ⋊ B"
        if z and not p and s and self.phi_trivial:
            return "2-cocycle Extension  A ⋊_ψ B"
        if z:
            return "Unified Product (cocycle double cross product)  A ⋈_ψ B"
        return "Bicocycle Double Cross Product (BDCP)  A_ζ ⋈_ψ B — full hierarchy"

    # ── the bracket itself ──────────────────────────────────────
    def expand(self, s1: BDCPSection, s2: BDCPSection, registry=None) -> BDCPSection:
        a1, b1 = s1.a, s1.b
        a2, b2 = s2.a, s2.b
        a_part = Sum(
            Sum(self.phi(a1, a2, registry), Neg(self.rho(b2, a1, registry))),
            Sum(self.rho(b1, a2, registry), self.psi(b1, b2, registry)),
        )
        b_part = Sum(
            Sum(self.theta(b1, b2, registry), Neg(self.sigma(a2, b1, registry))),
            Sum(self.sigma(a1, b2, registry), self.zeta(a1, a2, registry)),
        )
        return BDCPSection(a_part, b_part)

    def as_custom_bracket_on_flat_symbols(self, table: dict, name=None) -> CustomBracket:
        """
        For the SCALAR (finite-dimensional, one-basis-vector-per-slot)
        case used in the worked examples below, wrap a precomputed
        multiplication table into a Jacopy CustomBracket so that the
        existing, already-verified prove_jacobi engine can certify it
        directly — no new proof machinery required.
        """
        def bracket_fn(x, y, registry):
            return table.get((str(x), str(y)), Integer(0))
        return CustomBracket(
            name or self.name, bracket_fn,
            is_graded_antisymmetric=True,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=True,
        )


# Demo: build a BDCP bracket abstractly and expand it symbolically
print("\n  Abstract BDCP expansion (symbolic, no numbers):")
reg_demo = PropertyRegistry()
for s in (a1, b1, a2, b2):
    reg_demo.declare(s, Graded(degree=0))

def phi_demo(x, y, registry):   return Symbol(f"φ({to_ascii(x)},{to_ascii(y)})")
def zeta_demo(x, y, registry):  return Symbol(f"ζ({to_ascii(x)},{to_ascii(y)})")
def psi_demo(x, y, registry):   return Symbol(f"ψ({to_ascii(x)},{to_ascii(y)})")
def theta_demo(x, y, registry): return Symbol(f"θ({to_ascii(x)},{to_ascii(y)})")
def rho_demo(x, y, registry):   return Symbol(f"ρ({to_ascii(x)},{to_ascii(y)})")
def sigma_demo(x, y, registry): return Symbol(f"σ({to_ascii(x)},{to_ascii(y)})")

full_bdcp = BDCPBracket(phi_demo, zeta_demo, psi_demo, theta_demo, rho_demo, sigma_demo)
result = full_bdcp.expand(sec1, sec2, reg_demo)
print(f"  [sec1,sec2]_BDCP = {result}")
print(f"  classification   : {full_bdcp.classify()}")


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Worked instance: Heisenberg (2-cocycle → central extension)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3 — Heisenberg algebra as 2-cocycle extension")
print("=" * 60)

print("""
  g = span(Z) abelian, h = span(P,Q) abelian.
  Only nonzero map: psi(P,Q) = Z  (twisted 2-cocycle, h-side).
  This is g ⋊_psi h with rho = sigma = 0 too, i.e. Z is CENTRAL.
""")

reg_h = PropertyRegistry()
P = Symbol("P"); Q = Symbol("Q"); Z = Symbol("Z")
for s in (P, Q, Z):
    reg_h.declare(s, Graded(degree=0))

heisenberg_table = {
    ("P", "Q"): Z,          ("Q", "P"): Neg(Z),
    ("P", "Z"): Integer(0), ("Z", "P"): Integer(0),
    ("Q", "Z"): Integer(0), ("Z", "Q"): Integer(0),
    ("P", "P"): Integer(0), ("Q", "Q"): Integer(0), ("Z", "Z"): Integer(0),
}

def psi_heis(x, y, registry):
    return heisenberg_table.get((str(x), str(y)), Integer(0))

heisenberg_bdcp = BDCPBracket(psi=psi_heis, name="[·,·]_g⋈ψh")
print(f"  classification: {heisenberg_bdcp.classify()}")

bracket_heis = heisenberg_bdcp.as_custom_bracket_on_flat_symbols(
    heisenberg_table, name="[·,·]_Heisenberg"
)
try:
    chain_heis = prove_jacobi(bracket_heis, P, Q, Z, registry=reg_h)
    print(f"  ✓ Jacobi identity proved in {len(chain_heis)} step(s).")
except ProofFailure as e:
    print(f"  ✗ Jacobi failed: {e}")


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Worked instance: aff(1) (semi-direct product)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 4 — aff(1) = R ⋊ R as semi-direct product")
print("=" * 60)

print("""
  g = span(H), h = span(X), both abelian (phi = theta = 0).
  No twisted cocycles (psi = zeta = 0).
  Genuine one-sided action: rho(X,H) = X   (h acts on g), sigma = 0.
  This is the affine Lie algebra of the line.
""")

reg_a = PropertyRegistry()
H = Symbol("H"); X = Symbol("X")
for s in (H, X):
    reg_a.declare(s, Graded(degree=0))

aff1_table = {
    ("H", "X"): X, ("X", "H"): Neg(X),
    ("H", "H"): Integer(0), ("X", "X"): Integer(0),
}

def rho_aff1(x, y, registry):
    return aff1_table.get((str(y), str(x)), Integer(0))  # rho(b,a) convention

aff1_bdcp = BDCPBracket(rho=rho_aff1, name="[·,·]_A⋊B")
print(f"  classification: {aff1_bdcp.classify()}")

bracket_aff1 = aff1_bdcp.as_custom_bracket_on_flat_symbols(
    aff1_table, name="[·,·]_aff(1)"
)
try:
    # 2-dimensional algebra: repeat H to still exercise the cyclic sum
    chain_aff1 = prove_jacobi(bracket_aff1, H, X, H, registry=reg_a)
    print(f"  ✓ Jacobi identity proved in {len(chain_aff1)} step(s).")
except ProofFailure as e:
    print(f"  ✗ Jacobi failed: {e}")


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Hierarchy classification table (all 2^6 corners, pruned)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5 — Hierarchy classification (representative corners)")
print("=" * 60)

CORNERS = [
    ("all six nontrivial",        dict(phi=phi_demo, zeta=zeta_demo, psi=psi_demo,
                                        theta=theta_demo, rho=rho_demo, sigma=sigma_demo)),
    ("zeta=0 only",                dict(psi=psi_demo, phi=phi_demo, theta=theta_demo,
                                         rho=rho_demo, sigma=sigma_demo)),
    ("zeta=psi=0, rho,sigma live", dict(phi=phi_demo, theta=theta_demo,
                                         rho=rho_demo, sigma=sigma_demo)),
    ("zeta=psi=0, sigma only",     dict(phi=phi_demo, theta=theta_demo, sigma=sigma_demo)),
    ("zeta=0, sigma=0, phi=0",     dict(theta=theta_demo, psi=psi_demo)),
    ("everything trivial",         dict()),
]

for label, kwargs in CORNERS:
    b = BDCPBracket(**kwargs)
    print(f"  {label:32s} → {b.classify()}")


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Honest roadmap (what this module does NOT do)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 6 — What is still missing (not faked)")
print("=" * 60)
print("""
  This module handles the SCALAR / finite-basis case: every structure
  map is a Python function between symbolic generators, wrapped into
  a flat multiplication table for CustomBracket. This is exactly the
  level of generality needed to reproduce the paper's own concrete
  examples (Heisenberg, aff(1), and similarly-sized Lie algebras).

  NOT yet built (would need real additions to Jacopy's core, not
  just this tutorial file):

    1. Vector-bundle-valued structure maps — phi,zeta,psi,theta,rho,
       sigma acting on sections of genuine vector bundles A, B over
       a manifold M, rather than on a fixed finite basis. This needs
       BDCPSection to carry base-point / bundle-rank information the
       way SectionPair implicitly relies on TM ⊕ T*M's fixed rank.

    2. A contactization / z-extension rule (paper's eqs. 151-157) for
       the dissipative (Herglotz/contact) Hamilton's equations of §3,
       analogous to the K / K̃ Cartan-remainder operators already in
       15_intrinsic_engine.py and 17_tilde_calculus.py. Until this
       exists, the dissipative dynamics of the paper cannot be
       certified inside Jacopy — only the algebraic (Lie algebroid)
       layer, Thm 2.4/2.5, is currently in reach.

  Both additions are natural extensions of existing Jacopy machinery
  (CustomBracket for (1), the Cartan-remainder pattern for (2)) and
  would not require inventing a new proof paradigm.
""")

print("=" * 60)
print("32_bdcp_algebroids.py — ALL SECTIONS COMPLETE")
print("=" * 60)
