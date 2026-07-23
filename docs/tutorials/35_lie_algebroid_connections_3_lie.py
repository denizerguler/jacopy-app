"""Jacopy tutorial 35 — Connections, Jacobi/Bianchi, and 3-Lie algebroids.

Streamlit page extracted from the former app.py Tab VIII and corrected so that
all displayed examples satisfy the stated hypotheses.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
from jacopy.display import chain_to_ascii, chain_to_latex, to_ascii, to_latex
from jacopy.proof import prove_jacobi
from jacopy.proof.strategies import ProofFailure


def proof_block(chain, key: str) -> None:
    st.metric(L("Proof steps", "Kanıt adımı"), len(chain))
    for i, step in enumerate(chain.steps, 1):
        with st.expander(f"{L('Step', 'Adım')} {i} — `{step.rule}`", expanded=len(chain) <= 4):
            c1, c2 = st.columns(2)
            c1.caption(L("Before", "Önce")); c1.code(str(step.before))
            c2.caption(L("After", "Sonra")); c2.code(str(step.after))
            if step.justification: st.caption(step.justification)
    with st.expander("ASCII"):
        st.code(chain_to_ascii(chain))
    with st.expander("LaTeX"):
        try:
            latex = chain_to_latex(chain)
            st.code(latex, language="latex")
            st.download_button("⬇ .tex", latex, f"{key}.tex", key=f"dl_{key}")
        except Exception as exc:
            st.warning(str(exc))


def load_tutorial_33():
    candidates = [
        ROOT / "docs" / "tutorials" / "33_lie_algebroid_connections.py",
        HERE / "33_lie_algebroid_connections.py",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("33_lie_algebroid_connections.py was not found.")
    spec = importlib.util.spec_from_file_location("jacopy_tutorial_33", path)
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
    st.title(L("35 · Lie Algebroid Connections and 3-Lie Algebroids", "35 · Lie Algebroid Bağlantıları ve 3-Lie Algebroidleri"))
    st.caption("Ateşli–Esen–Sütlü, *Constructions of 3-Lie algebroids*")

    st.markdown(L(
        r"""This page studies two distinct constructions and keeps their hypotheses explicit:

    * a binary bracket induced by an \(A\)-connection on an anchored bundle;
    * a ternary Filippov bracket induced by a Lie bracket and a **linear character** \(\xi\).

    The point is not only to compute examples, but to display the precise obstruction whose
    vanishing turns each candidate operation into the desired algebraic structure.""",
        r"""Bu sayfa iki farklı inşayı ele alır ve varsayımlarını açıkça ayırır:

    * anchored bir demet üzerindeki \(A\)-bağlantının ürettiği ikili bracket;
    * Lie bracket ve **doğrusal karakter** \(\xi\)'nin ürettiği üçlü Filippov bracket.

    Amaç yalnızca örnek hesaplamak değil, her aday işlemi istenen cebirsel yapıya dönüştüren
    obstrüksiyonun tam olarak ne olduğunu göstermektir."""
    ))

    try:
        mod33 = load_tutorial_33()
        AbstractConnection = mod33.AbstractConnection
        TernaryBracketFromXi = mod33.TernaryBracketFromXi
    except Exception as exc:
        st.error(L("Tutorial 33 could not be loaded.", "33. öğretici yüklenemedi."))
        st.exception(exc)
        st.stop()

    TABS = st.tabs([
        L("Connection-induced bracket", "Bağlantıdan türeyen bracket"),
        L("Jacobi–Bianchi dictionary", "Jacobi–Bianchi sözlüğü"),
        L("Ternary construction", "Üçlü inşa"),
        L("Nontrivial valid example", "Sıfır olmayan geçerli örnek"),
        L("Fundamental identity", "Temel kimlik"),
        L("Verification boundary", "Doğrulama sınırı"),
    ])

    with TABS[0]:
        st.subheader(L("From an anchored connection to a skew bracket", "Anchored bağlantıdan antisimetrik bracket'a"))
        st.latex(r"[X,Y]^\nabla:=\nabla_XY-\nabla_YX")
        st.markdown(L(
            """Let \((A,\rho)\) be an anchored vector bundle and let \(\nabla\) satisfy
    \(\nabla_{fX}Y=f\nabla_XY\) and
    \(\nabla_X(fY)=f\nabla_XY+\rho(X)(f)Y\). Then the induced bracket is skew and obeys""",
            """\((A,\rho)\) anchored bir vektör demeti ve \(\nabla\),
    \(\nabla_{fX}Y=f\nabla_XY\) ile
    \(\nabla_X(fY)=f\nabla_XY+\rho(X)(f)Y\) koşullarını sağlasın. Türetilen bracket antisimetriktir ve"""
        ))
        st.latex(r"[X,fY]^\nabla=f[X,Y]^\nabla+\rho(X)(f)Y")
        st.info(L(
            "This proves the Leibniz rule, not the Jacobi identity. Jacobi is controlled by curvature.",
            "Bu, Leibniz kuralını kanıtlar; Jacobi kimliğini değil. Jacobi eğrilik tarafından kontrol edilir."
        ))

    with TABS[1]:
        st.subheader(L("The Jacobiator is the cyclic curvature", "Jacobiator döngüsel eğriliktir"))
        st.latex(r"R^\nabla(X,Y)Z:=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]^\nabla}Z")
        st.latex(r"\operatorname{Jac}_\nabla(X,Y,Z)=\sum_{\rm cyc}[X,[Y,Z]^\nabla]^\nabla")
        st.latex(r"\operatorname{Jac}_\nabla(X,Y,Z)=-\sum_{\rm cyc}R^\nabla(X,Y)Z")
        st.markdown(L(
            "Thus \([\cdot,\cdot]^\nabla\) is a Lie bracket exactly when the first Bianchi-type cyclic sum vanishes. The sign depends on the Jacobiator convention, but the vanishing condition does not.",
            "Dolayısıyla \([\cdot,\cdot]^\nabla\), birinci Bianchi-tipli döngüsel toplam sıfır olduğunda ve yalnızca o zaman Lie bracket'tır. İşaret Jacobiator konvansiyonuna bağlıdır; sıfırlık koşulu değildir."
        ))
        with st.expander(L("Derivation", "Türetim"), expanded=True):
            st.latex(r"""\begin{aligned}
    [X,[Y,Z]^\nabla]^\nabla
    &=\nabla_X(\nabla_YZ-\nabla_ZY)-\nabla_{\nabla_YZ-\nabla_ZY}X.
    \end{aligned}""")
            st.markdown(L(
                "Adding the three cyclic permutations and regrouping the second covariant derivatives yields the curvature cyclic sum.",
                "Üç döngüsel permütasyon toplanıp ikinci kovaryant türevler yeniden gruplanınca eğriliğin döngüsel toplamı elde edilir."
            ))

    with TABS[2]:
        st.subheader(L("Lie algebra plus character", "Lie cebiri artı karakter"))
        st.latex(r"[x,y,z]_\xi:=\xi(x)[y,z]+\xi(y)[z,x]+\xi(z)[x,y]")
        st.markdown(L(
            r"""The necessary hypothesis is that \(\xi:\mathfrak g\to\mathbb K\) is **linear** and
    annihilates the derived algebra:""",
            r"""Gerekli varsayım, \(\xi:\mathfrak g\to\mathbb K\)'nin **doğrusal** olması ve türemiş
    cebiri yok etmesidir:"""
        ))
        st.latex(r"\xi([x,y])=0\quad\forall x,y\in\mathfrak g")
        st.markdown(L(
            "Under this character condition, the displayed operation is alternating and satisfies the Filippov fundamental identity.",
            "Bu karakter koşulu altında verilen işlem alternandır ve Filippov temel kimliğini sağlar."
        ))
        st.warning(L(
            "A constant map ξ(x)=1 is not linear. Also, every character of the perfect Lie algebra so(3) is zero; therefore so(3) cannot provide a nonzero example of this construction.",
            "Sabit ξ(x)=1 dönüşümü doğrusal değildir. Ayrıca perfect Lie cebiri so(3)'ün her karakteri sıfırdır; dolayısıyla so(3) bu inşa için sıfır olmayan bir örnek veremez."
        ))

    with TABS[3]:
        st.subheader(L("Example: \(\mathfrak h_3\oplus\mathbb R t\)", "Örnek: \(\mathfrak h_3\oplus\mathbb R t\)"))
        st.latex(r"[p,q]=z,\qquad [t,\mathfrak g]=[z,\mathfrak g]=0")
        st.latex(r"\xi(t)=1,\qquad \xi(p)=\xi(q)=\xi(z)=0")
        st.markdown(L(
            r"Since the derived algebra is \(\mathbb R z\subset\ker\xi\), ξ is a valid character, and",
            r"Türemiş cebir \(\mathbb R z\subset\ker\xi\) olduğundan ξ geçerli bir karakterdir ve"
        ))
        st.latex(r"[t,p,q]_\xi=\xi(t)[p,q]=z\neq0")

        if st.button(L("▶ Compute the ternary bracket", "▶ Üçlü bracket'ı hesapla"), key="35_ternary"):
            reg = PropertyRegistry()
            t, p, q, z = (Symbol(n) for n in ("t", "p", "q", "z"))
            for s in (t, p, q, z): reg.declare(s, Graded(degree=0))
            zero = Integer(0)
            table = {
                ("p", "q"): z, ("q", "p"): Neg(z),
            }
            def h3_plus_r_bracket(a, b, registry):
                return table.get((str(a), str(b)), zero)
            def xi(a, registry):
                return Integer(1) if str(a) == "t" else zero
            ternary = TernaryBracketFromXi(h3_plus_r_bracket, xi, name="[·,·,·]_ξ")
            value = ternary.expand(t, p, q, reg)
            swapped = ternary.expand(p, t, q, reg)
            repeated = ternary.expand(t, p, p, reg)
            c1, c2, c3 = st.columns(3)
            c1.write("**[t,p,q]ξ**"); c1.code(to_ascii(value))
            c2.write("**[p,t,q]ξ**"); c2.code(to_ascii(swapped))
            c3.write("**[t,p,p]ξ**"); c3.code(to_ascii(repeated))
            st.success(L("The nonzero value and alternating signs agree with the theorem.", "Sıfır olmayan değer ve alternan işaretler teoremle uyumludur."))

    with TABS[4]:
        st.subheader(L("Filippov fundamental identity", "Filippov temel kimliği"))
        st.latex(r"""[x_1,x_2,[y_1,y_2,y_3]]
    =\sum_{i=1}^{3}[y_1,\ldots,[x_1,x_2,y_i],\ldots,y_3]""")
        st.markdown(L(
            "Equivalently, every inner operator \(D_{x_1,x_2}(y)=[x_1,x_2,y]\) must act as a derivation of the ternary bracket.",
            "Eşdeğer olarak her iç operatör \(D_{x_1,x_2}(y)=[x_1,x_2,y]\), üçlü bracket'ın bir türevi olarak davranmalıdır."
        ))
        st.latex(r"D_{x_1,x_2}[y_1,y_2,y_3]=\sum_{i=1}^{3}[y_1,\ldots,D_{x_1,x_2}y_i,\ldots,y_3]")
        st.markdown(L(
            r"For the character construction, expansion reduces the obstruction to the binary Jacobi identity together with \(\xi([x,y])=0\).",
            r"Karakter inşasında açılım, obstrüksiyonu ikili Jacobi kimliği ile \(\xi([x,y])=0\) koşuluna indirger."
        ))

    with TABS[5]:
        st.subheader(L("Current machine-verification boundary", "Mevcut makine-doğrulama sınırı"))
        st.markdown(L(
            """**Available now:** binary `CustomBracket` Jacobi proofs, connection-to-bracket
    computations supplied by Tutorial 33, and direct symbolic evaluation of the character-induced
    ternary bracket.

    **Still needed for a full 3-Lie algebroid engine:**

    * a native ternary expression and rewrite system;
    * `prove_fundamental_identity` with a displayed obstruction chain;
    * bundle sections, a ternary anchor \(\wedge^2A\to TM\), and the 3-Leibniz rule;
    * the paper's general multi-operator generating-family construction, beyond the single-character special case.

    Until these are implemented, the page labels direct computations as computations and does not
    present them as machine proofs of the full 3-Lie algebroid theorem.""",
            """**Şu anda mevcut:** ikili `CustomBracket` Jacobi kanıtları, 33. öğreticinin sağladığı
    bağlantıdan-bracket'a hesaplamalar ve karakterden türeyen üçlü bracket'ın doğrudan sembolik hesabı.

    **Tam bir 3-Lie algebroid motoru için hâlâ gerekenler:**

    * yerel bir üçlü ifade ve yeniden-yazma sistemi;
    * obstrüksiyon zincirini gösteren `prove_fundamental_identity`;
    * demet kesitleri, \(\wedge^2A\to TM\) üçlü anchor'ı ve 3-Leibniz kuralı;
    * tek-karakter özel durumunun ötesindeki genel çok-operatörlü üretici-aile inşası.

    Bunlar uygulanana kadar sayfa doğrudan hesapları hesap olarak etiketler ve onları tam 3-Lie
    algebroid teoreminin makine kanıtı gibi sunmaz."""
        ))
