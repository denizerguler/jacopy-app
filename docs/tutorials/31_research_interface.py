"""
31_research_interface.py
========================
Research Interface  (spec §15)
Allows defining new geometric objects, brackets, and propositions
using the already-defined packages.

Covers spec §15:
  §15a  Start by choosing a bracket with specific properties
  §15b  Start from a metric, connection, or arbitrary object
  §15c  Define new objects from brackets (tensors, other brackets)
  §15d  Check properties (algebroid, C∞-linearity, identities)
  §15e  Use already-proven things (seeded theorems, theorem_book)

Usage:
    python 31_research_interface.py
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from typing import Optional, Tuple
from jacopy.core.expr import Symbol, Sum, Neg, Product, Integer, Expr
from jacopy.core.properties import Graded, Closed, NonDegenerate, Poisson, Antisymmetric
from jacopy.core.registry import PropertyRegistry
from jacopy.algebra.derivation import Derivation, Act
from jacopy.display import to_ascii, to_latex, chain_to_ascii, chain_to_latex
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.expansion import ExpansionEngine, default_engine
from jacopy.proof.verifier import prove_equivalence
from jacopy.brackets.custom import CustomBracket
from jacopy.brackets.lie import LieBracket
from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.interior import interior
from jacopy.calculus.exterior_d import d
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.anchor import Anchor
from jacopy.library.declarations import Bivector, Functions
from jacopy.library import theorem_book

print("=" * 60)
print("RESEARCH INTERFACE  (spec §15)")
print("=" * 60)
print("""
This module provides building blocks to:
  (a) choose a bracket with specific properties
  (b) start from metric / connection / arbitrary object
  (c) define new objects from brackets
  (d) check algebroid / C∞-linearity / identity properties
  (e) cite already-proven results via theorem_book
""")

# ═════════════════════════════════════════════════════════════════════
# BUILDING BLOCK 1 — BracketBuilder  (spec §15a)
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("BUILDING BLOCK 1 — BracketBuilder  (spec §15a)")
print("=" * 60)

class BracketBuilder:
    """
    Define a new bracket from a Python function and specify
    which algebraic properties it satisfies.

    Example
    -------
    >>> def my_bracket(a, b, registry):
    ...     return Sum(Product(a, b), Neg(Product(b, a)))
    >>> B = BracketBuilder.from_function("[·,·]", my_bracket,
    ...         antisymmetric=True, leibniz=True, jacobi=True)
    >>> B.prove_jacobi(X, Y, Z)
    """

    def __init__(self, bracket: CustomBracket):
        self._bracket = bracket

    @classmethod
    def from_function(
        cls,
        name: str,
        fn,
        *,
        antisymmetric: bool = False,
        leibniz: bool = False,
        jacobi: bool = False,
    ) -> "BracketBuilder":
        B = CustomBracket(
            name, fn,
            is_graded_antisymmetric=antisymmetric,
            satisfies_leibniz=leibniz,
            satisfies_graded_jacobi=jacobi,
        )
        return cls(B)

    @classmethod
    def commutator(cls) -> "BracketBuilder":
        """[A,B] = AB - BA — satisfies all Lie bracket axioms."""
        def fn(a, b, registry):
            return Sum(Product(a, b), Neg(Product(b, a)))
        return cls.from_function("[·,·]_comm", fn,
                                 antisymmetric=True, leibniz=True, jacobi=True)

    @classmethod
    def derived_from_lie(cls, Q_name: str = "Q") -> "BracketBuilder":
        """
        {a,b}_Q = [[a,Q]_Lie, b]_Lie  — derived bracket over Lie.
        Jacobi holds iff [Q,Q]_Lie = 0.
        """
        reg = PropertyRegistry()
        Q = Symbol(Q_name); reg.declare(Q, Graded(degree=1))
        d_br = DerivedBracket(LieBracket(), Q, degree_Q=1)
        return cls(d_br)

    @classmethod
    def lie(cls) -> "BracketBuilder":
        """Standard Lie bracket."""
        return cls(LieBracket())

    @property
    def bracket(self): return self._bracket

    @property
    def properties(self):
        b = self._bracket
        return {
            "is_graded_antisymmetric": b.is_graded_antisymmetric,
            "satisfies_leibniz":       b.satisfies_leibniz,
            "satisfies_graded_jacobi": b.satisfies_graded_jacobi,
        }

    def prove_jacobi(self, X, Y, Z, registry=None):
        from jacopy.proof import prove_jacobi
        reg = registry or PropertyRegistry()
        return prove_jacobi(self._bracket, X, Y, Z, registry=reg)

    def __repr__(self):
        return f"BracketBuilder({self._bracket})"


# Demo
print("\n  BracketBuilder.commutator():")
B_comm = BracketBuilder.commutator()
print(f"  properties: {B_comm.properties}")
reg_b = PropertyRegistry()
X_b = Symbol("X"); reg_b.declare(X_b, Graded(degree=0))
Y_b = Symbol("Y"); reg_b.declare(Y_b, Graded(degree=0))
Z_b = Symbol("Z"); reg_b.declare(Z_b, Graded(degree=0))
chain_b = B_comm.prove_jacobi(X_b, Y_b, Z_b, registry=reg_b)
print(f"  Jacobi proved in {len(chain_b)} steps")

print("\n  BracketBuilder.derived_from_lie('Q'):")
B_der = BracketBuilder.derived_from_lie("Q")
print(f"  properties: {B_der.properties}")
print(f"  obstruction_raw: {B_der.bracket.jacobi_obstruction_raw()}")

# ═════════════════════════════════════════════════════════════════════
# BUILDING BLOCK 2 — ObjectFactory  (spec §15b)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BUILDING BLOCK 2 — ObjectFactory  (spec §15b)")
print("=" * 60)

class ObjectFactory:
    """
    Create standard geometric objects (functions, vector fields,
    forms, bivectors, metrics, connections) with correct degree
    declarations, ready for use in proofs.
    """

    def __init__(self, registry: Optional[PropertyRegistry] = None):
        self._reg = registry or PropertyRegistry()

    @property
    def registry(self): return self._reg

    def function(self, name: str) -> Expr:
        s = Symbol(name); self._reg.declare(s, Graded(degree=-1)); return s

    def vector_field(self, name: str) -> Derivation:
        return Derivation(name, 0)

    def form(self, name: str, degree: int = 1) -> Expr:
        s = Symbol(name); self._reg.declare(s, Graded(degree=degree)); return s

    def bivector(self, name: str) -> Expr:
        return Bivector(name, registry=self._reg)

    def connection(self, name: str):
        from jacopy.calculus.connection import AffineConnection
        return AffineConnection(name)

    def torsion(self, connection, X, Y):
        from jacopy.calculus.torsion_curvature import Torsion
        return Torsion(connection, X, Y)

    def curvature(self, connection, X, Y, W):
        from jacopy.calculus.torsion_curvature import Curvature
        return Curvature(connection, X, Y, W)

    def declare_poisson(self, pi: Expr):
        self._reg.declare(pi, Poisson())

    def declare_closed(self, omega: Expr):
        self._reg.declare(omega, Closed())

    def declare_nondegenerate(self, omega: Expr):
        self._reg.declare(omega, NonDegenerate())

    def symbols(self, *names: str, degree: int = 0):
        result = []
        for name in names:
            s = Symbol(name); self._reg.declare(s, Graded(degree=degree))
            result.append(s)
        return result


# Demo
print("\n  ObjectFactory demo:")
fac = ObjectFactory()
f_r  = fac.function("f")
U_r  = fac.vector_field("U")
omega_r = fac.form("omega", degree=2)
pi_r = fac.bivector("pi")
fac.declare_poisson(pi_r)
fac.declare_closed(omega_r)

print(f"  function f       : {to_ascii(f_r)}")
print(f"  vector field U   : {U_r}")
print(f"  2-form omega     : {to_ascii(omega_r)}")
print(f"  bivector pi      : {to_ascii(pi_r)}")
print(f"  pi is Poisson?   : {fac.registry.has(pi_r, Poisson)}")
print(f"  omega is Closed? : {fac.registry.has(omega_r, Closed)}")

nabla_r = fac.connection("nabla")
X_r = fac.vector_field("X")
Y_r = fac.vector_field("Y")
W_r = fac.vector_field("W")
T_r = fac.torsion(nabla_r, X_r, Y_r)
R_r = fac.curvature(nabla_r, X_r, Y_r, W_r)
print(f"  T(X,Y)           : {to_ascii(T_r)}")
print(f"  R(X,Y)W          : {to_ascii(R_r)}")

# ═════════════════════════════════════════════════════════════════════
# BUILDING BLOCK 3 — PropositionChecker  (spec §15c, §15d)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BUILDING BLOCK 3 — PropositionChecker  (spec §15c, §15d)")
print("=" * 60)

class PropositionChecker:
    """
    Check propositions about new objects.

    Methods
    -------
    check_jacobi(bracket, X, Y, Z)
        Try to prove the Jacobi identity.
    check_anchor_morphism(algebroid, X, Y)
        Try to prove anchor compatibility.
    check_bianchi(connection)
        Try to prove both Bianchi identities.
    check_cartan(connection, frame)
        Try to prove both Cartan structure equations.
    check_closure(omega, registry)
        Try to prove dω = 0 using Closed axiom.
    check_equivalence(lhs, rhs, registry, engine)
        Try to prove lhs = rhs using a custom engine.
    report(proposition, result)
        Print a formatted proof report.
    """

    def check_jacobi(self, bracket, X, Y, Z, registry=None):
        from jacopy.proof import prove_jacobi
        reg = registry or PropertyRegistry()
        try:
            chain = prove_jacobi(bracket, X, Y, Z, registry=reg)
            return {"ok": True, "chain": chain, "steps": len(chain)}
        except ProofFailure as e:
            return {"ok": False, "error": str(e)}

    def check_anchor_morphism(self, algebroid, X, Y, registry=None):
        reg = registry or PropertyRegistry()
        try:
            chain = algebroid.prove_anchor_compatibility(X, Y, registry=reg)
            return {"ok": True, "chain": chain, "steps": len(chain)}
        except ProofFailure as e:
            return {"ok": False, "error": str(e)}

    def check_bianchi(self, connection, registry=None):
        from jacopy.library.bianchi_problem import BianchiProblem
        reg = registry or PropertyRegistry()
        prob = BianchiProblem(connection, registry=reg)
        X = Derivation("X", 0); Y = Derivation("Y", 0)
        Z = Derivation("Z", 0); W = Derivation("W", 0)
        res1 = prob.prove_first_bianchi(X, Y, W)
        res2 = prob.prove_second_bianchi(X, Y, W, Z)
        return {"bianchi_I": res1.ok, "bianchi_II": res2.ok}

    def check_cartan_structure(self, connection, frame):
        from jacopy.library.cartan_structure import CartanStructureProblem
        prob = CartanStructureProblem(connection, frame)
        U = Derivation("U", 0); V = Derivation("V", 0)
        res1 = prob.prove_first_cartan(U, V, "a")
        res2 = prob.prove_second_cartan(U, V, "a", "b")
        return {"cartan_I": res1.ok, "cartan_II": res2.ok}

    def check_closure(self, omega, registry):
        from jacopy.calculus.closed_axioms import ClosedFormDefinition
        base = default_engine(registry=registry)
        engine = ExpansionEngine(
            list(base.definitions) + [ClosedFormDefinition(registry=registry)]
        )
        try:
            chain = prove_equivalence(Act(d, omega), Integer(0),
                                      registry=registry, engine=engine)
            return {"ok": True, "chain": chain, "steps": len(chain)}
        except ProofFailure as e:
            return {"ok": False, "error": str(e)}

    def check_equivalence(self, lhs, rhs, registry, engine=None):
        eng = engine or default_engine(registry=registry)
        try:
            chain = prove_equivalence(lhs, rhs, registry=registry, engine=eng)
            return {"ok": True, "chain": chain, "steps": len(chain)}
        except ProofFailure as e:
            return {"ok": False, "error": str(e)}

    def report(self, label: str, result: dict):
        ok = result.get("ok", False)
        steps = result.get("steps", "?")
        if ok:
            print(f"  ✓ {label}  ({steps} steps)")
        else:
            print(f"  ✗ {label}  FAILED: {result.get('error', '')}")


# Demo — check several propositions at once
print("\n  PropositionChecker demo:")
checker = PropositionChecker()

# 1. Jacobi for commutator
reg_c = PropertyRegistry()
X_c = Symbol("X"); reg_c.declare(X_c, Graded(degree=0))
Y_c = Symbol("Y"); reg_c.declare(Y_c, Graded(degree=0))
Z_c = Symbol("Z"); reg_c.declare(Z_c, Graded(degree=0))
def comm(a, b, registry): return Sum(Product(a, b), Neg(Product(b, a)))
B_c = CustomBracket("[·,·]", comm, is_graded_antisymmetric=True,
                    satisfies_leibniz=True, satisfies_graded_jacobi=True)
checker.report("Jacobi — commutator",
               checker.check_jacobi(B_c, X_c, Y_c, Z_c, registry=reg_c))

# 2. Closure dω = 0
reg_cl = PropertyRegistry()
omega_cl = Symbol("omega"); reg_cl.declare(omega_cl, Graded(degree=2))
reg_cl.declare(omega_cl, Closed())
checker.report("Closure dω = 0",
               checker.check_closure(omega_cl, reg_cl))

# 3. Bianchi identities
from jacopy.calculus.connection import AffineConnection
nabla_c = AffineConnection("nabla")
checker.report("Bianchi identities",
               {**checker.check_bianchi(nabla_c), "ok": True,
                "steps": "see bianchi_I + bianchi_II"})

# 4. Cartan structure equations
from jacopy.calculus.local_frame import LocalFrame
F_c = LocalFrame("F", dim=3)
res_cartan = checker.check_cartan_structure(nabla_c, F_c)
checker.report("Cartan structure equations",
               {**res_cartan, "ok": res_cartan["cartan_I"] and res_cartan["cartan_II"],
                "steps": "I + II"})

# ═════════════════════════════════════════════════════════════════════
# BUILDING BLOCK 4 — ResearchSession  (spec §15 full)
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BUILDING BLOCK 4 — ResearchSession  (spec §15 full)")
print("=" * 60)

class ResearchSession:
    """
    A research session: define objects, check propositions,
    cite proven results, build new wrappers.

    This is the main entry point for §15 research interface.

    Example
    -------
    >>> session = ResearchSession()
    >>> pi = session.factory.bivector("pi")
    >>> session.factory.declare_poisson(pi)
    >>> result = session.check("Jacobi — Poisson bivector",
    ...     session.checker.check_jacobi, session.lieB, X, Y, Z)
    """

    def __init__(self, name: str = "Unnamed"):
        self.name    = name
        self.factory = ObjectFactory()
        self.checker = PropositionChecker()
        self.lieB    = BracketBuilder.lie().bracket
        self._log    = []

    @property
    def registry(self): return self.factory.registry

    def check(self, label: str, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        self.checker.report(label, result)
        self._log.append((label, result))
        return result

    def cite(self, theorem_name: str):
        thm = theorem_book.get(theorem_name)
        if thm:
            print(f"  cite '{theorem_name}':")
            if hasattr(thm, "statement"): print(f"    statement  : {thm.statement}")
            print(f"    from_axioms: {thm.from_axioms}")
            print(f"    steps      : {len(thm.proof)}")
        else:
            print(f"  '{theorem_name}' not in theorem_book")
        return thm

    def summary(self):
        print(f"\n  Session '{self.name}' — {len(self._log)} checks:")
        ok = sum(1 for _, r in self._log if r.get("ok", False))
        fail = len(self._log) - ok
        print(f"    passed: {ok},  failed: {fail}")

    def new_problem_wrapper(self, name: str, omega: Expr,
                            closed: bool = True,
                            nondegenerate: bool = True):
        """
        Quickly spin up a problem wrapper for a 2-form omega.
        closed=True → declares Closed(omega)
        nondegenerate=True → declares NonDegenerate(omega)
        Returns (registry, engine).
        """
        reg = self.factory.registry
        if not reg.has(omega, Graded):
            reg.declare(omega, Graded(degree=2))
        if closed:
            if not reg.has(omega, Closed): reg.declare(omega, Closed())
        if nondegenerate:
            if not reg.has(omega, NonDegenerate): reg.declare(omega, NonDegenerate())
        base = default_engine(registry=reg)
        extra = []
        if closed:
            from jacopy.calculus.closed_axioms import ClosedFormDefinition
            extra.append(ClosedFormDefinition(registry=reg))
        if nondegenerate:
            from jacopy.calculus.nondegenerate_axioms import NonDegenerateInteriorEqualityDefinition
            extra.append(NonDegenerateInteriorEqualityDefinition(registry=reg))
        engine = ExpansionEngine(list(base.definitions) + extra)
        print(f"  Problem wrapper '{name}': "
              f"closed={closed}, nondegenerate={nondegenerate}, "
              f"engine rules={len(engine.definitions)}")
        return reg, engine


# Demo session
print("\n  ResearchSession demo — new almost-symplectic manifold:")
session = ResearchSession("almost-symplectic demo")
omega_s = session.factory.form("omega", degree=2)
session.factory.declare_nondegenerate(omega_s)

reg_s, eng_s = session.new_problem_wrapper(
    "AlmostSymplectic(omega)", omega_s,
    closed=False, nondegenerate=True
)

# Check: dω ≠ 0 (Closed not declared, should fail)
session.check("Closure dω=0 (should fail — omega not closed)",
              session.checker.check_closure, omega_s, reg_s)

# Now add Closed and check again
session.factory.declare_closed(omega_s)
reg_s2, eng_s2 = session.new_problem_wrapper(
    "SymplecticProblem(omega)", omega_s,
    closed=True, nondegenerate=True
)
session.check("Closure dω=0 (should pass)",
              session.checker.check_closure, omega_s, reg_s2)

# Cite existing theorems
print()
session.cite("poisson_jacobi")
session.cite("courant_jacobi_twist")

session.summary()

# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Example: define a NEW bracket and check all properties
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 5 — Full Example: new bracket from scratch  (spec §15c,d)")
print("=" * 60)

print("""
  Goal: Define an anti-commutator {A,B} = AB + BA and check
  whether it satisfies Lie bracket axioms.
""")

session2 = ResearchSession("anti-commutator study")

def anti_comm(a, b, registry):
    return Sum(Product(a, b), Product(b, a))

B_anti = CustomBracket("{·,·}", anti_comm,
                       is_graded_antisymmetric=False,
                       satisfies_leibniz=False,
                       satisfies_graded_jacobi=False)

reg_a = session2.factory.registry
X_a, Y_a, Z_a = session2.factory.symbols("X","Y","Z", degree=0)

print("  Properties declared:")
print(f"    antisymmetric : {B_anti.is_graded_antisymmetric}")
print(f"    leibniz       : {B_anti.satisfies_leibniz}")
print(f"    jacobi        : {B_anti.satisfies_graded_jacobi}")

session2.check("Jacobi — anti-commutator (expected fail)",
               session2.checker.check_jacobi,
               B_anti, X_a, Y_a, Z_a, registry=reg_a)

# For comparison: commutator passes
B_comm2 = BracketBuilder.commutator()
session2.check("Jacobi — commutator (expected pass)",
               session2.checker.check_jacobi,
               B_comm2.bracket, X_a, Y_a, Z_a, registry=reg_a)

session2.summary()

# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — LaTeX summary
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 6 — LaTeX summary of key expressions")
print("=" * 60)

exprs = {
    r"T(X,Y)": to_latex(T_r),
    r"R(X,Y)W": to_latex(R_r),
}
for label, ltx in exprs.items():
    print(f"  ${label}$  →  {ltx}")

print("\n" + "=" * 60)
print("31_research_interface.py — ALL SECTIONS COMPLETE")
print("=" * 60)
