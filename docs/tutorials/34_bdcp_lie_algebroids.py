"""Jacopy tutorial 34 — Bicocycle Double Cross Product Lie algebroids.

Streamlit page extracted from the former app.py Tab VII.
The page separates definitions, compatibility data, degeneration hierarchy,
and machine-verifiable finite-dimensional examples.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import streamlit as st

HERE = Path(__file__).resolve().parent
for candidate in (HERE, Path.cwd().resolve(), *Path.cwd().resolve().parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        ROOT = candidate
        break
else:
    ROOT = Path.cwd().resolve()


from jacopy.core.expr import Integer, Neg, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.display import chain_to_ascii, chain_to_latex, to_ascii
from jacopy.proof import prove_jacobi
from jacopy.proof.strategies import ProofFailure


def proof_block(chain, key: str) -> None:
    st.metric(L("Proof steps", "Kanıt adımı"), len(chain))
    for i, step in enumerate(chain.steps, 1):
        with st.expander(f"{L('Step', 'Adım')} {i} — `{step.rule}`", expanded=len(chain) <= 4):
            c1, c2 = st.columns(2)
            c1.caption(L("Before", "Önce")); c1.code(str(step.before))
            c2.caption(L("After", "Sonra")); c2.code(str(step.after))
            if step.justification:
                st.caption(step.justification)
    c1, c2 = st.columns(2)
    with c1.expander("ASCII"):
        st.code(chain_to_ascii(chain))
    with c2.expander("LaTeX"):
        try:
            latex = chain_to_latex(chain)
            st.code(latex, language="latex")
            st.download_button("⬇ .tex", latex, f"{key}.tex", key=f"dl_{key}")
        except Exception as exc:
            st.warning(str(exc))


def load_tutorial_32():
    candidates = [
        ROOT / "docs" / "tutorials" / "32_bdcp_algebroids.py",
        HERE / "32_bdcp_algebroids.py",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("32_bdcp_algebroids.py was not found.")
    spec = importlib.util.spec_from_file_location("jacopy_tutorial_32", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render(L, TR=False):
    """Render this tutorial inside the parent Jacopy Streamlit application."""
    globals()["L"] = L
    globals()["TR"] = TR
    st.title(L("34 · Bicocycle Double Cross Product Lie Algebroids", "34 · Bikosiklik Çift Çapraz Çarpım Lie Algebroidleri"))
    st.caption("Ateşli–Esen–Sütlü, *On Product Lie Algebroids, and Collective Motion*")

    st.markdown(L(
        """Let \(A\to M\) and \(B\to M\) be anchored vector bundles. A BDCP structure packages
    six bilinear maps into a skew bracket on \(A\oplus B\). The construction is not merely a
    formula: the resulting bracket defines a Lie algebroid only when its Jacobiator and anchor
    obstruction vanish. This page therefore separates **raw structure data** from the
    **compatibility equations** that turn that data into geometry.""",
        """\(A\to M\) ve \(B\to M\) anchored vektör demetleri olsun. BDCP yapısı altı bilineer
    dönüşümü \(A\oplus B\) üzerinde antisimetrik bir bracket içinde birleştirir. Bu yalnızca bir
    formül değildir: elde edilen bracket ancak Jacobiator ve anchor obstrüksiyonu sıfır olduğunda
    bir Lie algebroid tanımlar. Bu sayfa bu nedenle **ham yapı verisini**, onu geometriye dönüştüren
    **uyumluluk denklemlerinden** ayırır."""
    ))

    st.latex(r"""\begin{aligned}
    [(a_1,b_1),(a_2,b_2)]
    =\big(&\phi(a_1,a_2)+\rho(b_1,a_2)-\rho(b_2,a_1)+\psi(b_1,b_2),\\
    &\theta(b_1,b_2)+\sigma(a_1,b_2)-\sigma(a_2,b_1)+\zeta(a_1,a_2)\big).
    \end{aligned}""")

    with st.expander(L("Precise type signature", "Kesin tip bilgisi"), expanded=True):
        st.latex(r"\phi:\Gamma(A)^2\to\Gamma(A),\quad \zeta:\Gamma(A)^2\to\Gamma(B)")
        st.latex(r"\theta:\Gamma(B)^2\to\Gamma(B),\quad \psi:\Gamma(B)^2\to\Gamma(A)")
        st.latex(r"""\rho:\Gamma(B)\times\Gamma(A)\to\Gamma(A),\quad
                  \sigma:\Gamma(A)\times\Gamma(B)\to\Gamma(B)""")
        st.markdown(L(
            "Here \(\phi,\theta\) are internal brackets, \(\rho,\sigma\) are mutual actions, and \(\psi,\zeta\) are cocycle terms.",
            "Burada \(\phi,\theta\) iç bracket'lar, \(\rho,\sigma\) karşılıklı etkiler, \(\psi,\zeta\) ise kosiklik terimleridir."
        ))

    st.markdown("---")
    try:
        mod32 = load_tutorial_32()
        BDCPSection = mod32.BDCPSection
        BDCPBracket = mod32.BDCPBracket
    except Exception as exc:
        st.error(L("Tutorial 32 could not be loaded.", "32. öğretici yüklenemedi."))
        st.exception(exc)
        st.stop()

    TABS = st.tabs([
        L("Definition & obstructions", "Tanım ve obstrüksiyonlar"),
        L("Degeneration hierarchy", "Dejenerasyon hiyerarşisi"),
        L("Heisenberg extension", "Heisenberg genişlemesi"),
        L("Semidirect product", "Yarı-direkt çarpım"),
        L("Scope of verification", "Doğrulamanın kapsamı"),
    ])

    with TABS[0]:
        st.subheader(L("What must be proved?", "Ne kanıtlanmalıdır?"))
        st.markdown(L(
            """For sections \(u_i=(a_i,b_i)\), define
    \(J(u_1,u_2,u_3)=\sum_{\rm cyc}[u_1,[u_2,u_3]]\). A genuine Lie algebroid requires:

    1. skew-symmetry of the total bracket;
    2. \(J=0\);
    3. the Leibniz rule with total anchor \(\alpha_A+\alpha_B\);
    4. equivalently, anchor morphism compatibility once Jacobi and Leibniz are established.

    Expanding \(J\) into its \(A\)- and \(B\)-components produces the matched-pair and cocycle
    compatibility equations. Jacopy's current finite-basis layer verifies the resulting total
    bracket directly rather than pretending to prove the full bundle-valued theorem.""",
            """\(u_i=(a_i,b_i)\) kesitleri için
    \(J(u_1,u_2,u_3)=\sum_{\rm cyc}[u_1,[u_2,u_3]]\) tanımlansın. Gerçek bir Lie algebroid için:

    1. toplam bracket antisimetrik olmalı;
    2. \(J=0\) olmalı;
    3. toplam anchor \(\alpha_A+\alpha_B\) ile Leibniz kuralı sağlanmalı;
    4. Jacobi ve Leibniz kurulduğunda anchor morfizmi uyumluluğu elde edilmelidir.

    \(J\)'nin \(A\)- ve \(B\)-bileşenlerine açılması matched-pair ve kosiklik uyumluluk
    denklemlerini verir. Jacopy'nin mevcut sonlu-baz katmanı, tam demet-değerli teoremi
    kanıtladığını iddia etmek yerine oluşan toplam bracket'ı doğrudan doğrular."""
        ))
        st.latex(r"J_{A\oplus B}=0\iff \operatorname{pr}_A J=0\ \text{and}\ \operatorname{pr}_B J=0")

    with TABS[1]:
        st.subheader(L("Which maps are switched on?", "Hangi dönüşümler açık?"))
        maps = {
            "φ": st.checkbox("φ", True, key="34_phi"),
            "θ": st.checkbox("θ", True, key="34_theta"),
            "ρ": st.checkbox("ρ", True, key="34_rho"),
            "σ": st.checkbox("σ", True, key="34_sigma"),
            "ψ": st.checkbox("ψ", True, key="34_psi"),
            "ζ": st.checkbox("ζ", True, key="34_zeta"),
        }
        active = {k for k, v in maps.items() if v}
        if active == {"φ", "θ", "ρ", "σ", "ψ", "ζ"}:
            kind = "BDCP"
        elif active <= {"φ", "θ", "ρ", "σ"} and {"ρ", "σ"} & active:
            kind = "double cross / matched pair"
        elif active <= {"φ", "θ", "ρ"} or active <= {"φ", "θ", "σ"}:
            kind = "semidirect-type"
        elif active <= {"φ", "θ", "ψ", "ζ"} and {"ψ", "ζ"} & active:
            kind = "cocycle extension-type"
        elif active <= {"φ", "θ"}:
            kind = "direct product"
        else:
            kind = "mixed degeneration"
        st.info(f"{L('Structural class', 'Yapısal sınıf')}: **{kind}**")
        st.caption(L(
            "This is a structural classification only; Jacobi still imposes equations on the nonzero maps.",
            "Bu yalnızca yapısal bir sınıflandırmadır; Jacobi sıfır olmayan dönüşümlere hâlâ denklemler dayatır."
        ))

    with TABS[2]:
        st.subheader(L("Central 2-cocycle extension", "Merkezi 2-kosiklik genişleme"))
        st.latex(r"[x,y]_{\omega}=[x,y]_{\mathfrak g}+\omega(x,y)c,\qquad [c,\cdot]=0")
        st.markdown(L(
            "The Jacobi identity is equivalent to the Chevalley–Eilenberg cocycle condition \(d_{CE}\omega=0\).",
            "Jacobi kimliği Chevalley–Eilenberg kosiklik koşulu \(d_{CE}\omega=0\)'a denktir."
        ))
        st.latex(r"(d_{CE}\omega)(x,y,z)=\omega([x,y],z)+\omega([y,z],x)+\omega([z,x],y)=0")
        if st.button(L("▶ Verify the 3D Heisenberg bracket", "▶ 3B Heisenberg bracket'ını doğrula"), key="34_heis"):
            reg = PropertyRegistry()
            x, y, c = (Symbol(n) for n in ("x", "y", "c"))
            for s in (x, y, c): reg.declare(s, Graded(degree=0))
            table = {
                ("x", "y"): c, ("y", "x"): Neg(c),
                ("x", "c"): Integer(0), ("c", "x"): Integer(0),
                ("y", "c"): Integer(0), ("c", "y"): Integer(0),
                ("x", "x"): Integer(0), ("y", "y"): Integer(0), ("c", "c"): Integer(0),
            }
            bdcp = BDCPBracket(name="Heisenberg central extension")
            bracket = bdcp.as_custom_bracket_on_flat_symbols(table, name="[·,·]_h3")
            try:
                chain = prove_jacobi(bracket, x, y, c, registry=reg)
                st.success(L("Jacobi verified.", "Jacobi doğrulandı."))
                proof_block(chain, "34_heisenberg")
            except ProofFailure as exc:
                st.error(str(exc))

    with TABS[3]:
        st.subheader(L("Semidirect product example: \(\mathfrak{aff}(1)\)", "Yarı-direkt çarpım örneği: \(\mathfrak{aff}(1)\)"))
        st.latex(r"[H,X]=X,\qquad [H,H]=[X,X]=0")
        st.markdown(L(
            "This is \(\mathbb R H\ltimes\mathbb R X\), where \(H\) acts on the abelian ideal by the identity derivation.",
            "Bu, \(H\)'nin abelyen ideal üzerinde özdeşlik türeviyle etki ettiği \(\mathbb R H\ltimes\mathbb R X\)'tir."
        ))
        if st.button(L("▶ Verify Jacobi", "▶ Jacobi'yi doğrula"), key="34_aff"):
            reg = PropertyRegistry(); H, X = Symbol("H"), Symbol("X")
            for s in (H, X): reg.declare(s, Graded(degree=0))
            table = {
                ("H", "X"): X, ("X", "H"): Neg(X),
                ("H", "H"): Integer(0), ("X", "X"): Integer(0),
            }
            bdcp = BDCPBracket(name="aff(1) semidirect product")
            bracket = bdcp.as_custom_bracket_on_flat_symbols(table, name="[·,·]_aff(1)")
            try:
                chain = prove_jacobi(bracket, H, X, H, registry=reg)
                st.success(L("Jacobi verified.", "Jacobi doğrulandı."))
                proof_block(chain, "34_aff1")
            except ProofFailure as exc:
                st.error(str(exc))

    with TABS[4]:
        st.subheader(L("What this page proves—and what it does not", "Bu sayfa neyi kanıtlıyor, neyi kanıtlamıyor?"))
        st.markdown(L(
            """**Currently machine-verified:** finite-dimensional structure tables converted to Jacopy
    `CustomBracket` objects, followed by symbolic Jacobi reduction.

    **Not yet machine-verified in full generality:** bundle-valued \(C^\infty(M)\)-bilinearity,
    all component compatibility identities for six arbitrary maps, and the total-anchor Leibniz
    rule on genuine sections. Those require native typed sections of \(A\oplus B\), anchors, and a
    componentwise obstruction engine.""",
            """**Şu anda makine tarafından doğrulanan:** sonlu boyutlu yapı tablolarının Jacopy
    `CustomBracket` nesnelerine çevrilmesi ve ardından sembolik Jacobi indirgemesi.

    **Henüz tam genellikte doğrulanmayan:** demet-değerli \(C^\infty(M)\)-bilineerlik, altı keyfi
    dönüşümün tüm bileşen uyumlulukları ve gerçek kesitlerde toplam-anchor Leibniz kuralı. Bunlar
    \(A\oplus B\)'nin tipli kesitlerini, anchor'ları ve bileşen bazlı bir obstrüksiyon motorunu gerektirir."""
        ))
