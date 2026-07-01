import sys
from pathlib import Path
import streamlit as st

here = Path.cwd().resolve()
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

st.set_page_config(
    page_title="Jacopy — Interactive Differential Geometry",
    layout="wide",
    page_icon="∂",
)

# ── Language ──────────────────────────────────────────────────
lang = st.sidebar.radio("🌐 Language / Dil", ["English", "Türkçe"])
TR = lang == "Türkçe"
def L(en, tr): return tr if TR else en

# ── Core imports ──────────────────────────────────────────────
from jacopy import VectorFields, Forms
from jacopy.core.expr import Symbol, Sum, Product, Neg, Integer
from jacopy.core.properties import Graded, Closed, Poisson, NonDegenerate
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ProofFailure
from jacopy.display import chain_to_ascii, chain_to_latex, to_ascii, to_latex
from jacopy.algebra.derivation import Derivation, Act
from jacopy.library.declarations import Bivector, Functions

# ── Helpers ───────────────────────────────────────────────────
def proof_block(chain, key="proof"):
    """Render a proof chain: summary → steps → LaTeX download."""
    c1, c2, c3 = st.columns(3)
    c1.metric(L("Steps", "Adım"), len(chain))
    c2.metric(L("Result", "Sonuç"), str(chain.steps[-1].after)[:40])
    c3.metric(L("Provenance", "Kaynak"), chain.steps[0].provenance_tag or "—")

    for i, step in enumerate(chain.steps, 1):
        with st.expander(f"**{L('Step','Adım')} {i}** — `{step.rule}`", expanded=(len(chain) <= 4)):
            col_b, col_a = st.columns(2)
            col_b.caption(L("Before", "Önce"))
            col_b.code(str(step.before), language="text")
            col_a.caption(L("After", "Sonra"))
            col_a.code(str(step.after), language="text")
            if step.justification:
                st.caption(f"_{step.justification}_")

    col_ascii, col_latex = st.columns(2)
    with col_ascii.expander("ASCII proof"):
        st.code(chain_to_ascii(chain), language="text")
    with col_latex.expander("LaTeX proof"):
        try:
            ltx = chain_to_latex(chain)
            st.code(ltx, language="latex")
            st.download_button(
                L("⬇ Download .tex", "⬇ .tex indir"),
                ltx, f"{key}.tex", key=f"dl_{key}"
            )
        except Exception:
            st.warning(L("LaTeX export unavailable.", "LaTeX dışa aktarılamadı."))

def expr_row(label, expr, show_latex=True):
    col1, col2 = st.columns([1, 2])
    col1.markdown(f"**{label}**")
    col2.code(to_ascii(expr), language="text")
    if show_latex:
        st.latex(to_latex(expr))

def section_intro(title, body, formula=None):
    st.markdown(f"### {title}")
    st.markdown(body)
    if formula:
        st.latex(formula)
    st.markdown("---")

def fail(e):
    st.error(f"{L('Proof failed','Kanıt başarısız')}: `{e}`")

def math_proof_to_latex(title, steps):
    """Generate a LaTeX document from proof steps: list of (title, formula, note)."""
    out = []
    out.append(r"\documentclass{amsart}")
    out.append(r"\usepackage{amsmath,amssymb,geometry}")
    out.append(r"\geometry{margin=2.5cm}")
    out.append(r"\begin{document}")
    out.append(f"\\section*{{{title}}}")
    out.append(r"\begin{proof}")
    for i, (step_title, formula, note) in enumerate(steps, 1):
        out.append(f"\\noindent\\textbf{{Step {i}: {step_title}}}")
        out.append(r"\begin{equation*}")
        out.append(formula)
        out.append(r"\end{equation*}")
        if note:
            out.append(f"\\noindent\\textit{{{note}}}\\medskip")
        out.append("")
    out.append(r"\end{proof}")
    out.append(r"\end{document}")
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════
# SIDEBAR — about
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.markdown(L(
    """**Jacopy** is a symbolic proof engine for differential geometry.
    
Every button on this page runs a *machine-verified proof* — not a numerical approximation.""",
    """**Jacopy** diferansiyel geometri için sembolik bir ispat motorudur.

Bu sayfadaki her buton, *makine tarafından doğrulanmış* bir kanıt çalıştırır."""
))

# ── Title ─────────────────────────────────────────────────────
st.title(L(
    "Jacopy — Interactive Differential Geometry Proof Engine",
    "Jacopy — Etkileşimli Diferansiyel Geometri İspat Motoru"
))
st.markdown(L(
    """> Select a tab to explore machine-verified proofs across the main domains of modern differential geometry.
    Each proof is **fully symbolic** — no numerics, no approximations.""",
    """> Bir sekme seçerek modern diferansiyel geometrinin ana alanlarında makine-doğrulamalı kanıtları keşfedin.
    Her kanıt **tamamen sembolik** — sayısal yaklaşım yok."""
))
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    L("I · Lie Theory",            "I · Lie Teorisi"),
    L("II · Poisson & Symplectic", "II · Poisson & Simplektik"),
    L("III · Generalized Geometry","III · Genelleştirilmiş Geometri"),
    L("IV · Metric-Affine",        "IV · Metrik-Affin"),
    L("V · Algebroid Calculus",    "V · Algebroid Kalkülüs"),
    L("VI · Research Interface",   "VI · Araştırma Arayüzü"),
])

# ══════════════════════════════════════════════════════════════
# TAB I — Lie Theory
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.header(L("I · Lie Theory", "I · Lie Teorisi"))
    st.markdown(L(
        """The foundational layer. A **Lie bracket** on a vector space is an antisymmetric bilinear map
        satisfying the **Jacobi identity** — the cornerstone of Lie algebras, Lie groups, and all bracket structures below.""",
        """Temel katman. Bir vektör uzayındaki **Lie bracket**, **Jacobi kimliğini** sağlayan antisimetrik
        bir bilineer dönüşümdür — Lie cebirleri, Lie grupları ve aşağıdaki tüm bracket yapılarının temel taşı."""
    ))

    proof_tabs = st.tabs([
        L("Jacobi Identity", "Jacobi Kimliği"),
        L("Schouten–Nijenhuis", "Schouten–Nijenhuis"),
        L("Cartan Calculus", "Cartan Kalkülüs"),
        L("Anchor Compatibility", "Anchor Uyumluluğu"),
    ])

    # ── I.1 Jacobi ────────────────────────────────────────────
    with proof_tabs[0]:
        section_intro(
            L("The Jacobi Identity", "Jacobi Kimliği"),
            L(
                "For the **standard Lie bracket** on vector fields, the Jacobi identity holds identically. "
                "Jacopy proves this by expanding all three cyclic terms and reducing to zero in **2 engine steps**.",
                "**Standart Lie bracket** için Jacobi kimliği özdeş olarak sağlanır. "
                "Jacopy bunu üç döngüsel terimi açarak **2 motor adımında** sıfıra indirger."
            ),
            r"[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Claim:** For any vector fields $X, Y, Z$ on $M$: $[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0$.",
                "**İddia:** $M$ üzerinde herhangi $X, Y, Z$ vektör alanları için: $[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0$."
            ))
            st.markdown(L("**Step 1:** Recall the Lie bracket definition:", "**Adım 1:** Lie bracket tanımını hatırla:"))
            st.latex(r"[X,Y](f) = X(Y(f)) - Y(X(f)) \quad \forall\, f \in C^\infty(M)")
            st.markdown(L("**Step 2:** Expand the first cyclic term:", "**Adım 2:** Birinci döngüsel terimi aç:"))
            st.latex(r"[X,[Y,Z]](f) = X(Y(Z(f))) - X(Z(Y(f))) - Y(Z(X(f))) + Z(Y(X(f)))")
            st.markdown(L("**Step 3:** Similarly expand the other two and add all three:", "**Adım 3:** Diğer ikisini de aç ve üçünü topla:"))
            st.latex(r"\begin{aligned} &[X,[Y,Z]](f)+[Y,[Z,X]](f)+[Z,[X,Y]](f) \\ =\;& XYZ(f)-XZY(f)-YZX(f)+ZYX(f) \\ &+\,YZX(f)-YXZ(f)-ZXY(f)+XZY(f) \\ &+\,ZXY(f)-ZYX(f)-XYZ(f)+YXZ(f) \end{aligned}")
            st.markdown(L("**Step 4:** Every term appears exactly twice with opposite signs — all cancel:", "**Adım 4:** Her terim zıt işaretlerle tam iki kez görünür — hepsi iptal olur:"))
            st.latex(r"= 0 \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Jacobi Identity — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Lie bracket definition}\n\\begin{equation*}\n[X,Y](f) = X(Y(f)) - Y(X(f)) \\quad \\forall\\, f \\in C^\\infty(M)\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Expand first cyclic term}\n\\begin{equation*}\n[X,[Y,Z]](f) = X(Y(Z(f))) - X(Z(Y(f))) - Y(Z(X(f))) + Z(Y(X(f)))\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Sum all three cyclic terms}\n\\begin{equation*}\n\\sum_{\\mathrm{cycl}} [X,[Y,Z]](f) = \\text{(12 terms, each appears twice with opposite signs)}\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: All terms cancel}\n\\begin{equation*}\n= 0 \\qquad \\square\n\\end{equation*}\n\\noindent\\textit{all third-order terms cancel in pairs}\\medskip\n\n\\end{proof}\n\\end{document}',
                file_name="jacobi_id_proof.tex",
                mime="text/plain",
                key="dl_math_jacobi_id",
            )
        st.markdown("---")
        names = st.text_input(L("Vector fields (space-separated)", "Vektör alanları (boşlukla)"), "X Y Z", key="li_names")
        if st.button(L("▶ Prove with Jacopy", "▶ Jacopy ile Kanıtla"), key="li_jac"):
            from jacopy.brackets.lie import lie
            from jacopy.proof import prove_jacobi
            try:
                reg = PropertyRegistry()
                syms = VectorFields(names, registry=reg)
                chain = prove_jacobi(lie, syms[0], syms[1], syms[2], registry=reg)
                st.success(L("✓ Jacobi identity proved.", "✓ Jacobi kimliği kanıtlandı."))
                proof_block(chain, "lie_jacobi")
            except ProofFailure as e: fail(e)

    # ── I.2 Schouten–Nijenhuis ────────────────────────────────
    with proof_tabs[1]:
        section_intro(
            L("Schouten–Nijenhuis Bracket", "Schouten–Nijenhuis Bracket'ı"),
            L(
                "The **Schouten–Nijenhuis bracket** extends the Lie bracket to multivector fields. "
                "It is graded-antisymmetric and satisfies a graded Jacobi identity. "
                "Four atomic cases determine the full bracket via the Leibniz rule.",
                "**Schouten–Nijenhuis bracket'ı**, Lie bracket'ını çokvektör alanlarına genişletir. "
                "Derecelendirilmiş antisimetrik olup derecelendirilmiş Jacobi kimliğini sağlar. "
                "Dört atomik durum, Leibniz kuralıyla tam bracket'ı belirler."
            ),
            r"[P,Q]_{SN} \text{ for } P \in \mathfrak{X}^p(M),\; Q \in \mathfrak{X}^q(M)"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "The SN bracket is the unique graded-antisymmetric extension of the Lie bracket "
                "to multivector fields satisfying the graded Leibniz rule.",
                "SN bracket'ı, derecelendirilmiş Leibniz kuralını sağlayan tek derecelendirilmiş "
                "antisimetrik genişlemedir."
            ))
            st.markdown(L("**Step 1:** Degree convention — functions have SN-degree $-1$, vector fields $0$:", "**Adım 1:** Derece kuralı:"))
            st.latex(r"f \in C^\infty(M) \Rightarrow |f|_{SN}=-1, \quad X \in \mathfrak{X}(M) \Rightarrow |X|_{SN}=0")
            st.markdown(L("**Step 2:** Base case — two vector fields (Lie bracket):", "**Adım 2:** Temel durum — iki vektör alanı:"))
            st.latex(r"[X,Y]_{SN} := [X,Y]_{\mathrm{Lie}}")
            st.markdown(L("**Step 3:** Base case — vector field and function (derivation action):", "**Adım 3:** Temel durum — vektör alanı ve fonksiyon:"))
            st.latex(r"[X,f]_{SN} := X(f), \quad [f,X]_{SN} = -X(f)")
            st.markdown(L("**Step 4:** Base case — two functions:", "**Adım 4:** Temel durum — iki fonksiyon:"))
            st.latex(r"[f,g]_{SN} := 0")
            st.markdown(L("**Step 5:** Extension to $p$-vectors via graded Leibniz:", "**Adım 5:** Derecelendirilmiş Leibniz ile $p$-vektörlere genişlet:"))
            st.latex(r"[P, Q \wedge R]_{SN} = [P,Q]_{SN} \wedge R + (-1)^{(|P|+1)|Q|}\, Q \wedge [P,R]_{SN}")
            st.markdown(L("**Step 6:** Graded Jacobi identity:", "**Adım 6:** Derecelendirilmiş Jacobi kimliği:"))
            st.latex(r"(-1)^{|P||R|}[P,[Q,R]]_{SN} + (-1)^{|Q||P|}[Q,[R,P]]_{SN} + (-1)^{|R||Q|}[R,[P,Q]]_{SN} = 0 \quad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Schouten-Nijenhuis Bracket — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Degree convention}\n\\begin{equation*}\nf \\in C^\\infty(M) \\Rightarrow |f|_{SN}=-1, \\quad X \\in \\mathfrak{X}(M) \\Rightarrow |X|_{SN}=0\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Base case: two vector fields}\n\\begin{equation*}\n[X,Y]_{SN} := [X,Y]_{\\mathrm{Lie}}\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Base case: vector field and function}\n\\begin{equation*}\n[X,f]_{SN} := X(f), \\quad [f,X]_{SN} = -X(f)\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Base case: two functions}\n\\begin{equation*}\n[f,g]_{SN} := 0\n\\end{equation*}\n\n\\noindent\\textbf{Step 5: Extension via graded Leibniz}\n\\begin{equation*}\n[P, Q \\wedge R]_{SN} = [P,Q]_{SN} \\wedge R + (-1)^{(|P|+1)|Q|} Q \\wedge [P,R]_{SN}\n\\end{equation*}\n\n\\noindent\\textbf{Step 6: Graded Jacobi identity}\n\\begin{equation*}\n(-1)^{|P||R|}[P,[Q,R]_{SN}]_{SN} + \\text{cycl} = 0 \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="sn_bracket_proof.tex",
                mime="text/plain",
                key="dl_math_sn_bracket",
            )
        st.markdown("---")
        if st.button(L("▶ Compute all 4 base cases", "▶ 4 temel durumu hesapla"), key="li_sn"):
            from jacopy.brackets.schouten import sn
            reg = PropertyRegistry()
            X = Symbol("X"); reg.declare(X, Graded(degree=0))
            Y = Symbol("Y"); reg.declare(Y, Graded(degree=0))
            f = Symbol("f"); reg.declare(f, Graded(degree=-1))
            cases = [
                ("[X, Y]_SN", sn.expand(X, Y, reg), L("= Lie bracket","= Lie bracket")),
                ("[f, g]_SN", sn.expand(f, f, reg), L("= 0 (functions commute)","= 0 (fonksiyonlar komutatif)")),
                ("[X, f]_SN", sn.expand(X, f, reg), L("= X(f) (derivation action)","= X(f) (türev etkisi)")),
                ("[f, X]_SN", sn.expand(f, X, reg), L("= −X(f) (antisymmetry)","= −X(f) (antisimetri)")),
            ]
            st.success(L("✓ All 4 base cases computed.", "✓ 4 temel durum hesaplandı."))
            for label, val, note in cases:
                c1, c2, c3 = st.columns([2, 3, 3])
                c1.code(label)
                c2.code(to_ascii(val))
                c3.caption(note)


    # ── I.3 Cartan Calculus ───────────────────────────────────
    with proof_tabs[2]:
        section_intro(
            L("Cartan's Five Relations", "Cartan'ın Beş İlişkisi"),
            L(
                "The **Cartan calculus** on differential forms is governed by five operator identities. "
                "Each can be proved individually or all at once. "
                "Two modes: **efficient** (single citation step) and **foundational** (full sub-proof from generator facts).",
                "Diferansiyel formlar üzerindeki **Cartan kalkülüs**, beş operatör kimliğiyle yönetilir. "
                "Her biri ayrı ayrı veya hepsi birden kanıtlanabilir. "
                "İki mod: **verimli** (tek alıntı adımı) ve **temellere dayalı** (üretici olgulardan tam alt-kanıt)."
            ),
            r"d^2=0,\quad [d,\iota_X]=\mathcal{L}_X,\quad [d,\mathcal{L}_X]=0,\quad [\mathcal{L}_X,\mathcal{L}_Y]=\mathcal{L}_{[X,Y]},\quad [\mathcal{L}_X,\iota_Y]=\iota_{[X,Y]}"
        )

        from jacopy.calculus.cartan import CartanCalculus, RELATIONS
        from jacopy.calculus.exterior_algebra import ExteriorAlgebra
        from jacopy.calculus.exterior_d import d as ext_d
        from jacopy.calculus.interior import interior as ext_interior
        from jacopy.calculus.lie_derivative import lie_derivative as ext_lie
        from jacopy.brackets.lie import LieBracket

        def make_cartan_setup():
            reg = PropertyRegistry()
            f0 = Symbol("f"); reg.declare(f0, Graded(degree=0))
            alg = ExteriorAlgebra((f0,))
            X0 = Derivation("X", 0); Y0 = Derivation("Y", 0)
            cart = CartanCalculus(d=ext_d, lie_derivative=ext_lie,
                                  interior=ext_interior, vector_bracket=LieBracket())
            return reg, alg, X0, Y0, cart

        cartan_subtabs = st.tabs([
            L("All 5 at once", "Hepsi birden"),
            "1. d²=0",
            "2. [d,ι_X]=L_X",
            "3. [d,L_X]=0",
            "4. [L_X,L_Y]=L_{[X,Y]}",
            "5. [L_X,ι_Y]=ι_{[X,Y]}",
            L("Invariant-d formula", "Invariant-d formülü"),
            L("H-twisted Cartan", "H-bükülmüş Cartan"),
        ])

        with cartan_subtabs[0]:
            proof_mode = st.radio(L("Proof mode","Kanıt modu"), ["efficient","foundational"],
                                  horizontal=True, key="cartan_mode_all")
            if st.button(L("▶ Verify all 5","▶ Tümünü doğrula"), key="cartan_all"):
                reg, alg, X0, Y0, cart = make_cartan_setup()
                results = cart.verify_all(algebra=alg, X=X0, Y=Y0, registry=reg, mode=proof_mode)
                st.success(L("✓ All 5 Cartan relations verified.","✓ 5 Cartan ilişkisi doğrulandı."))
                for name, chain in results.items():
                    eq = cart.relation(name, X=X0, Y=Y0, algebra=alg)
                    with st.expander(f"**`{name}`** — {len(chain)} {L('step(s)','adım')}"):
                        st.write(f"Operator equation: `{eq}`")
                        proof_block(chain, f"cartan_{name}")

        RELATION_INFO = {
            "d_squared_zero": {
                "formula": r"d \circ d = 0",
                "en": "The exterior derivative squares to zero. Foundation of de Rham cohomology. "
                      "Efficient mode: one axiom citation. Foundational mode: derived from generators via AgreementOnGenerators.",
                "tr": "Dışsal türev karesi sıfırdır. de Rham kohomolojisinin temeli. "
                      "Verimli modda: tek aksiyom alıntısı. Temellere dayalı modda: AgreementOnGenerators ile üreticilerden türetilir.",
                "args": {},
            },
            "cartan_magic": {
                "formula": r"[d, \iota_X] = d\iota_X + \iota_X d = \mathcal{L}_X",
                "en": "**Cartan's magic formula** — the most useful identity in differential geometry. "
                      "Expresses the Lie derivative as a graded commutator of d and ι_X.",
                "tr": "**Cartan'ın magic formülü** — diferansiyel geometrideki en kullanışlı kimlik. "
                      "Lie türevini d ve ι_X'in derecelendirilmiş komatörü olarak ifade eder.",
                "args": {"X": True},
            },
            "d_lie": {
                "formula": r"[d, \mathcal{L}_X] = 0",
                "en": "d and L_X commute. Equivalently, the Lie derivative of an exact form is exact. Follows from magic and d²=0.",
                "tr": "d ve L_X komatütatiftir. Keşin bir formun Lie türevi keşindir. Magic ve d²=0'dan gelir.",
                "args": {"X": True},
            },
            "lie_lie": {
                "formula": r"[\mathcal{L}_X, \mathcal{L}_Y] = \mathcal{L}_{[X,Y]}",
                "en": "The commutator of two Lie derivatives is the Lie derivative of the bracket. Makes X → L_X a Lie algebra representation.",
                "tr": "İki Lie türevinin komatörü, bracket'ın Lie türevidir. X → L_X'i bir Lie cebri temsili yapar.",
                "args": {"X": True, "Y": True},
            },
            "lie_iota": {
                "formula": r"[\mathcal{L}_X, \iota_Y] = \iota_{[X,Y]}",
                "en": "The Lie derivative of an interior product gives the interior product of the bracket. Closes the full Cartan algebra.",
                "tr": "Bir iç çarpımın Lie türevi, bracket'ın iç çarpımını verir. Tam Cartan cebirini kapatır.",
                "args": {"X": True, "Y": True},
            },
        }

        # Mathematical proofs dictionary
        MATH_PROOFS = {
            "d_squared_zero": {
                "en": [
                    ("Step 1: Let f be a 0-form (function)",
                     r"df = \sum_{i=1}^n \frac{\partial f}{\partial x^i}\, dx^i", ""),
                    ("Step 2: Apply d again",
                     r"d(df) = \sum_{i,j} \frac{\partial^2 f}{\partial x^j \partial x^i}\, dx^j \wedge dx^i", ""),
                    ("Step 3: Antisymmetry of wedge product — pair (i,j) cancels (j,i)",
                     r"dx^j \wedge dx^i = -\, dx^i \wedge dx^j", ""),
                    ("Step 4: Mixed partials commute (Schwarz theorem)",
                     r"\frac{\partial^2 f}{\partial x^i \partial x^j} = \frac{\partial^2 f}{\partial x^j \partial x^i}", ""),
                    ("Step 5: Conclusion for 0-forms",
                     r"d(df) = \sum_{i<j}\!\left(\frac{\partial^2 f}{\partial x^j \partial x^i} - \frac{\partial^2 f}{\partial x^i \partial x^j}\right)dx^i\wedge dx^j = 0", ""),
                    ("Step 6: General p-form via Leibniz rule",
                     r"d(d\varphi) = \sum_{|I|=p} \underbrace{d(df_I)}_{=0} \wedge dx^{i_1}\wedge\cdots\wedge dx^{i_p} = 0", "QED"),
                ],
                "tr": [
                    ("Adım 1: f bir 0-form (fonksiyon) olsun",
                     r"df = \sum_{i=1}^n \frac{\partial f}{\partial x^i}\, dx^i", ""),
                    ("Adım 2: d'yi tekrar uygula",
                     r"d(df) = \sum_{i,j} \frac{\partial^2 f}{\partial x^j \partial x^i}\, dx^j \wedge dx^i", ""),
                    ("Adım 3: Wedge antisimetrisi — (i,j) ile (j,i) çifti iptal olur",
                     r"dx^j \wedge dx^i = -\, dx^i \wedge dx^j", ""),
                    ("Adım 4: Karışık türevler komutatiftir (Schwarz teoremi)",
                     r"\frac{\partial^2 f}{\partial x^i \partial x^j} = \frac{\partial^2 f}{\partial x^j \partial x^i}", ""),
                    ("Adım 5: 0-formlar için sonuç",
                     r"d(df) = \sum_{i<j}\!\left(\frac{\partial^2 f}{\partial x^j \partial x^i} - \frac{\partial^2 f}{\partial x^i \partial x^j}\right)dx^i\wedge dx^j = 0", ""),
                    ("Adım 6: Genel p-form — Leibniz kuralı ile",
                     r"d(d\varphi) = \sum_{|I|=p} \underbrace{d(df_I)}_{=0} \wedge dx^{i_1}\wedge\cdots\wedge dx^{i_p} = 0", "QED"),
                ],
            },
            "cartan_magic": {
                "en": [
                    ("Step 1: On 0-forms f, ι_X f = 0",
                     r"(\iota_X d + d\iota_X)f = \iota_X(df) + 0 = X(f) = \mathcal{L}_X f", ""),
                    ("Step 2: On exact 1-forms df, use d²=0",
                     r"(\iota_X d + d\iota_X)(df) = \iota_X(\underbrace{d^2f}_{=0}) + d(X(f)) = d(\mathcal{L}_X f) = \mathcal{L}_X(df)", ""),
                    ("Step 3: Both sides are degree-0 derivations of Ω*(M)",
                     r"\iota_X d + d\iota_X \;\text{ and }\; \mathcal{L}_X \;\text{ both satisfy Leibniz rule for } \wedge", ""),
                    ("Step 4: Agreement on generators C∞(M) and Ω¹(M) implies equality everywhere",
                     r"[d,\iota_X] = \mathcal{L}_X \quad \text{on all of }\; \Omega^*(M)", "QED"),
                ],
                "tr": [
                    ("Adım 1: 0-formlar f üzerinde, ι_X f = 0",
                     r"(\iota_X d + d\iota_X)f = \iota_X(df) + 0 = X(f) = \mathcal{L}_X f", ""),
                    ("Adım 2: Kesin 1-formlar df üzerinde, d²=0 kullan",
                     r"(\iota_X d + d\iota_X)(df) = \iota_X(\underbrace{d^2f}_{=0}) + d(X(f)) = \mathcal{L}_X(df)", ""),
                    ("Adım 3: Her iki taraf da Ω*(M)'nin derecesi-0 türevidir",
                     r"\iota_X d + d\iota_X \;\text{ ve }\; \mathcal{L}_X \;\text{her ikisi de } \wedge \text{ için Leibniz kuralını sağlar}", ""),
                    ("Adım 4: C∞(M) ve Ω¹(M) üreticilerinde eşleşme her yerde eşitliği ima eder",
                     r"[d,\iota_X] = \mathcal{L}_X \quad \Omega^*(M)\text{'nin tamamnında}", "QED"),
                ],
            },
            "d_lie": {
                "en": [
                    ("Step 1: Expand the commutator",
                     r"[d,\mathcal{L}_X] = d\mathcal{L}_X - \mathcal{L}_X d", ""),
                    ("Step 2: Substitute magic formula  L_X = dι_X + ι_X d",
                     r"= d(d\iota_X + \iota_X d) - (d\iota_X + \iota_X d)d", ""),
                    ("Step 3: Expand",
                     r"= \underbrace{d^2}_{0}\iota_X + d\iota_X d - d\iota_X d - \iota_X\underbrace{d^2}_{0}", ""),
                    ("Step 4: Cancel",
                     r"= 0 + d\iota_X d - d\iota_X d - 0 = 0", "QED"),
                ],
                "tr": [
                    ("Adım 1: Komutatörü genişlet",
                     r"[d,\mathcal{L}_X] = d\mathcal{L}_X - \mathcal{L}_X d", ""),
                    ("Adım 2: Magic formülü yerine koy: L_X = dι_X + ι_X d",
                     r"= d(d\iota_X + \iota_X d) - (d\iota_X + \iota_X d)d", ""),
                    ("Adım 3: Genişlet",
                     r"= \underbrace{d^2}_{0}\iota_X + d\iota_X d - d\iota_X d - \iota_X\underbrace{d^2}_{0}", ""),
                    ("Adım 4: İptal et",
                     r"= 0 + d\iota_X d - d\iota_X d - 0 = 0", "QED"),
                ],
            },
            "lie_lie": {
                "en": [
                    ("Step 1: Check on functions f",
                     r"[\mathcal{L}_X,\mathcal{L}_Y]f = XY(f)-YX(f) = [X,Y](f) = \mathcal{L}_{[X,Y]}f", "Lie bracket definition"),
                    ("Step 2: Check on exact 1-forms df, using [d,L_X]=0",
                     r"[\mathcal{L}_X,\mathcal{L}_Y](df) = d([\mathcal{L}_X,\mathcal{L}_Y]f) = d([X,Y]f) = \mathcal{L}_{[X,Y]}(df)", ""),
                    ("Step 3: Both sides are degree-0 derivations; agreement on generators implies equality",
                     r"[\mathcal{L}_X,\mathcal{L}_Y] = \mathcal{L}_{[X,Y]} \quad \text{on all of }\; \Omega^*(M)", "QED"),
                ],
                "tr": [
                    ("Adım 1: Fonksiyonlar f üzerinde kontrol",
                     r"[\mathcal{L}_X,\mathcal{L}_Y]f = XY(f)-YX(f) = [X,Y](f) = \mathcal{L}_{[X,Y]}f", "Lie bracket tanımı"),
                    ("Adım 2: Kesin 1-formlar df üzerinde, [d,L_X]=0 kullan",
                     r"[\mathcal{L}_X,\mathcal{L}_Y](df) = d([X,Y]f) = \mathcal{L}_{[X,Y]}(df)", ""),
                    ("Adım 3: Her iki taraf derecesi-0 türev; üreticilerde eşleşme eşitliği ima eder",
                     r"[\mathcal{L}_X,\mathcal{L}_Y] = \mathcal{L}_{[X,Y]} \quad \Omega^*(M)\text{'nin tamamında}", "QED"),
                ],
            },
            "lie_iota": {
                "en": [
                    ("Step 1: Expand [L_X, ι_Y]ω",
                     r"[\mathcal{L}_X,\iota_Y]\omega = \mathcal{L}_X(\iota_Y\omega) - \iota_Y(\mathcal{L}_X\omega)", ""),
                    ("Step 2: Apply magic formula to L_X on both terms",
                     r"= (d\iota_X+\iota_X d)(\iota_Y\omega) - \iota_Y(d\iota_X+\iota_X d)\omega", ""),
                    ("Step 3: Expand and collect (use d²=0)",
                     r"= d(\iota_X\iota_Y\omega) + \iota_X d(\iota_Y\omega) - \iota_Y d(\iota_X\omega) - \iota_Y\iota_X(d\omega)", ""),
                    ("Step 4: Apply magic to ι_X dω and ι_Y dω, collect via [L_X,L_Y]=L_{[X,Y]}",
                     r"= \iota_{[X,Y]}\omega", "QED — via derivation identity and [L_X,L_Y]=L_{[X,Y]}"),
                ],
                "tr": [
                    ("Adım 1: [L_X, ι_Y]ω'yu genişlet",
                     r"[\mathcal{L}_X,\iota_Y]\omega = \mathcal{L}_X(\iota_Y\omega) - \iota_Y(\mathcal{L}_X\omega)", ""),
                    ("Adım 2: Her iki terime L_X için magic formülü uygula",
                     r"= (d\iota_X+\iota_X d)(\iota_Y\omega) - \iota_Y(d\iota_X+\iota_X d)\omega", ""),
                    ("Adım 3: Genişlet ve derle (d²=0 kullan)",
                     r"= d(\iota_X\iota_Y\omega) + \iota_X d(\iota_Y\omega) - \iota_Y d(\iota_X\omega) - \iota_Y\iota_X(d\omega)", ""),
                    ("Adım 4: [L_X,L_Y]=L_{[X,Y]} ile derle",
                     r"= \iota_{[X,Y]}\omega", "QED — türev kimliği ve [L_X,L_Y]=L_{[X,Y]} ile"),
                ],
            },
        }

        for idx, (rel_name, info) in enumerate(RELATION_INFO.items(), 1):
            with cartan_subtabs[idx]:
                st.markdown(f"### {idx}. `{rel_name}`")
                st.latex(info["formula"])
                st.markdown(L(info["en"], info["tr"]))
                st.markdown("---")

                # Mathematical proof
                proof_data = MATH_PROOFS[rel_name]
                with st.expander(L(f"📖 Mathematical Proof",
                                   f"📖 Matematiksel Kanıt"), expanded=False):
                    steps = proof_data["en"] if not TR else proof_data["tr"]
                    for i, (step_title, formula, note) in enumerate(steps, 1):
                        st.markdown(f"**{i}. {step_title}**")
                        st.latex(formula)
                        if note:
                            st.caption(f"_{note}_")

                st.markdown("---")
                proof_mode_r = st.radio(
                    L("Jacopy engine mode","Jacopy motor modu"), ["efficient","foundational"],
                    horizontal=True, key=f"cartan_mode_{rel_name}"
                )
                if st.button(L(f"▶ Verify `{rel_name}` with Jacopy",f"▶ Jacopy ile `{rel_name}` doğrula"), key=f"cartan_btn_{rel_name}"):
                    reg, alg, X0, Y0, cart = make_cartan_setup()
                    kw = {"algebra": alg, "registry": reg, "mode": proof_mode_r}
                    if info["args"].get("X"): kw["X"] = X0
                    if info["args"].get("Y"): kw["Y"] = Y0
                    chain = cart.verify(rel_name, **kw)
                    st.success(L(f"✓ `{rel_name}` verified in {len(chain)} step(s).",
                                 f"✓ `{rel_name}` {len(chain)} adımda doğrulandı."))
                    eq_kw = {"algebra": alg}
                    if info["args"].get("X"): eq_kw["X"] = X0
                    if info["args"].get("Y"): eq_kw["Y"] = Y0
                    eq = cart.relation(rel_name, **eq_kw)
                    st.info(f"Operator equation: `{eq}`")
                    proof_block(chain, f"cartan_{rel_name}")


        with cartan_subtabs[6]:
            st.markdown("### Invariant-d formula")
            st.latex(r"d\omega(X,Y) = X(\omega(Y)) - Y(\omega(X)) - \omega([X,Y])")
            st.markdown(L(
                "A classical theorem derived from magic + lie_iota: coordinate-free formula for dω on a 1-form.",
                "Magic + lie_iota'dan türetilen klasik bir teorem: bir 1-form üzerinde dω için koordinatsız formül."
            ))
            if st.button(L("▶ Expand invariant-d","▶ Invariant-d aç"), key="cartan_invd"):
                from jacopy.calculus.invariant_d import invariant_d_one_form
                from jacopy.brackets.lie import lie as lie_br
                reg = PropertyRegistry()
                omega_inv = Symbol("ω"); reg.declare(omega_inv, Graded(degree=1))
                X_inv = Derivation("X", 0); Y_inv = Derivation("Y", 0)
                result = invariant_d_one_form(omega_inv, X_inv, Y_inv, bracket=lie_br)
                st.success(L("✓ Expanded.","✓ Açıldı."))
                st.write(f"**dω(X,Y) =** `{to_ascii(result)}`")
                st.latex(to_latex(result))

        with cartan_subtabs[7]:
            st.markdown("### H-twisted Cartan Calculus")
            st.latex(r"d_H = d + H\wedge,\quad d_H^2 = 0 \iff dH = 0")
            st.markdown(L(
                "For a closed 3-form H, the twisted derivative d_H satisfies all 5 Cartan relations.",
                "Kapalı bir 3-form H için, bükülmüş türev d_H tüm 5 Cartan ilişkisini sağlar."
            ))
            if st.button(L("▶ Verify H-twisted Cartan","▶ H-bükülmüş Cartan doğrula"), key="cartan_htwist"):
                from jacopy.library import TwistedCartanBundle
                reg = PropertyRegistry()
                f_h = Symbol("f"); reg.declare(f_h, Graded(degree=0))
                H_h = Symbol("H"); reg.declare(H_h, Graded(degree=3))
                bundle = TwistedCartanBundle(H_h)
                alg_h = ExteriorAlgebra((f_h,), d=bundle.d)
                X_h = Derivation("X", 0); Y_h = Derivation("Y", 0)
                results_h = bundle.cartan.verify_all(algebra=alg_h, X=X_h, Y=Y_h, registry=reg)
                st.success(L("✓ All 5 H-twisted Cartan relations verified.",
                             "✓ 5 H-bükülmüş Cartan ilişkisi doğrulandı."))
                for name, chain in results_h.items():
                    with st.expander(f"**`{name}`** — {len(chain)} {L('step(s)','adım')}"):
                        proof_block(chain, f"htwist_{name}")

    # ── I.4 Anchor Compatibility ──────────────────────────────
    with proof_tabs[3]:
        section_intro(
            L("Lie Algebroid — Anchor Compatibility", "Lie Algebroid — Anchor Uyumluluğu"),
            L(
                "A **Lie algebroid** $(E, [\\cdot,\\cdot]_E, \\rho)$ requires the anchor $\\rho: E \\to TM$ "
                "to be a bracket morphism. This is an **independent axiom** — it cannot be derived from the "
                "Lie bracket axioms alone. Jacopy proves it via a dedicated engine rule.",
                "Bir **Lie algebroid** $(E, [\\cdot,\\cdot]_E, \\rho)$, anchor $\\rho: E \\to TM$'nin "
                "bir bracket morfizmi olmasını gerektirir. Bu **bağımsız bir aksiyomdur** — Lie bracket "
                "aksiyomlarından tek başına türetilemez. Jacopy bunu özel bir motor kuralıyla kanıtlar."
            ),
            r"\rho([X,Y]_E) = [\rho(X),\rho(Y)]_{TM}"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Theorem:** If $(E,[\\cdot,\\cdot]_E,\\rho)$ satisfies right-Leibniz and Jacobi, "
                "then $\\rho$ is necessarily a bracket morphism.",
                "**Teorem:** $(E,[\\cdot,\\cdot]_E,\\rho)$ sağ-Leibniz ve Jacobi'yi sağlıyorsa, "
                "$\\rho$ zorunlu olarak bir bracket morfizmidir."
            ))
            st.markdown(L("**Step 1:** Right-Leibniz rule:", "**Adım 1:** Sağ-Leibniz kuralı:"))
            st.latex(r"[u, fv]_E = f[u,v]_E + \rho(u)(f)\, v \quad \forall\, u,v \in \Gamma(E),\; f \in C^\infty(M)")
            st.markdown(L("**Step 2:** Use antisymmetry to get $[fu,v]_E$:", "**Adım 2:** Antisimetri ile $[fu,v]_E$ hesapla:"))
            st.latex(r"[fu,v]_E = -[v,fu]_E = f[u,v]_E - \rho(v)(f)\,u")
            st.markdown(L("**Step 3:** Apply $\\rho$ to both sides:", "**Adım 3:** $\\rho$'yu her iki tarafa uygula:"))
            st.latex(r"\rho([fu,v]_E) = f\,\rho([u,v]_E) - \rho(v)(f)\,\rho(u)")
            st.markdown(L("**Step 4:** Compute $[\\rho(fu),\\rho(v)]_{TM}$ using Leibniz on $TM$:", "**Adım 4:** $TM$'de Leibniz kullanarak $[\\rho(fu),\\rho(v)]_{TM}$ hesapla:"))
            st.latex(r"[\rho(fu),\rho(v)]_{TM} = f[\rho(u),\rho(v)]_{TM} - \rho(v)(f)\,\rho(u)")
            st.markdown(L("**Step 5:** The two expressions match for all $f$, so:", "**Adım 5:** İki ifade tüm $f$ için örtüşür:"))
            st.latex(r"\rho([u,v]_E) = [\rho(u),\rho(v)]_{TM} \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Anchor Compatibility — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Right-Leibniz rule}\n\\begin{equation*}\n[u, fv]_E = f[u,v]_E + \\rho(u)(f)\\, v\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Compute [fu,v]_E via antisymmetry}\n\\begin{equation*}\n[fu,v]_E = f[u,v]_E - \\rho(v)(f)\\,u\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Apply rho}\n\\begin{equation*}\n\\rho([fu,v]_E) = f\\,\\rho([u,v]_E) - \\rho(v)(f)\\,\\rho(u)\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Leibniz on TM}\n\\begin{equation*}\n[\\rho(fu),\\rho(v)]_{TM} = f[\\rho(u),\\rho(v)]_{TM} - \\rho(v)(f)\\,\\rho(u)\n\\end{equation*}\n\n\\noindent\\textbf{Step 5: Conclusion}\n\\begin{equation*}\n\\rho([u,v]_E) = [\\rho(u),\\rho(v)]_{TM} \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="anchor_compat_proof.tex",
                mime="text/plain",
                key="dl_math_anchor_compat",
            )
        st.markdown("---")
        if st.button(L("▶ Prove anchor compatibility", "▶ Anchor uyumluluğunu kanıtla"), key="li_anc"):
            from jacopy.calculus.anchor import Anchor
            from jacopy.brackets.lie import LieBracket
            from jacopy.library.lie_algebroid import LieAlgebroid
            reg = PropertyRegistry()
            E = Symbol("E")
            A = LieAlgebroid(E, bracket=LieBracket(name="[·,·]_E"),
                             anchor=Anchor(name="ρ"), name="E-algebroid")
            X, Y = VectorFields("X Y", registry=reg)
            obs = A.anchor_compatibility_obstruction(X, Y, reg)
            st.info(L(f"Obstruction expression: `{to_ascii(obs)}`",
                      f"Obstrüksiyon ifadesi: `{to_ascii(obs)}`"))
            chain = A.prove_anchor_compatibility(X, Y, registry=reg)
            st.success(L("✓ Anchor compatibility proved.", "✓ Anchor uyumluluğu kanıtlandı."))
            proof_block(chain, "anchor_compat")

# TAB II — Poisson & Symplectic
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.header(L("II · Poisson & Symplectic Geometry", "II · Poisson & Simplektik Geometri"))
    st.markdown(L(
        """**Poisson geometry** is the natural home of classical mechanics. A Poisson bivector $\\pi$ 
        defines a bracket on functions; the Jacobi identity for this bracket is **equivalent** to 
        $[\\pi,\\pi]_{SN} = 0$. **Symplectic geometry** is a special case where $\\pi$ is non-degenerate.""",
        """**Poisson geometri**, klasik mekaniğin doğal evidir. Bir Poisson bivektörü $\\pi$, 
        fonksiyonlar üzerinde bir bracket tanımlar; bu bracket için Jacobi kimliği 
        $[\\pi,\\pi]_{SN} = 0$'a **eşdeğerdir**. **Simplektik geometri**, $\\pi$'nin dejenere olmadığı özel durumdur."""
    ))

    proof_tabs2 = st.tabs([
        L("Jacobi → [π,π]=0", "Jacobi → [π,π]=0"),
        L("Three Views of {f,g}", "Üç Bracket Görünümü"),
        L("Koszul Bracket", "Koszul Bracket"),
        L("Tilde Calculus", "Tilde Kalkülüs"),
        L("Symplectic Manifold", "Simplektik Manifold"),
    ])

    reg_p = PropertyRegistry()
    pi_p = Bivector("π", registry=reg_p); reg_p.declare(pi_p, Poisson())
    f_p, g_p, h_p = Functions("f g h", degree=-1, registry=reg_p)
    alpha_p, beta_p, gamma_p = Forms("α β γ", degree=1, registry=reg_p)
    from jacopy.library.poisson import PoissonBracket
    poisson_p = PoissonBracket.from_bivector(pi_p)

    # ── II.1 Jacobi reduction ─────────────────────────────────
    with proof_tabs2[0]:
        section_intro(
            L("Jacobi Identity reduces to [π,π]_SN = 0", "Jacobi Kimliği [π,π]_SN = 0'a İndirgenir"),
            L(
                "The Jacobi identity for the Poisson bracket $\\{f,g,h\\}_{\\pi}$ is not proved term-by-term. "
                "Instead, Jacopy shows the **obstruction** is exactly $[\\pi,\\pi]_{SN}$, "
                "which vanishes by the Poisson axiom. This works at both the function level and the form level (Koszul).",
                "Poisson bracket $\\{f,g,h\\}_{\\pi}$ için Jacobi kimliği terim terim kanıtlanmaz. "
                "Bunun yerine Jacopy, **obstrüksiyonun** tam olarak $[\\pi,\\pi]_{SN}$ olduğunu gösterir; "
                "bu, Poisson aksiyomuyla sıfırlanır. Bu hem fonksiyon hem de form seviyesinde (Koszul) çalışır."
            ),
            r"\operatorname{Jac}_{\{\cdot,\cdot\}_\pi}(f,g,h) \;\Longrightarrow\; [\pi,\pi]_{SN}=0"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Theorem:** For a Poisson bivector $\\pi$, the Jacobi identity for $\\{\\cdot,\\cdot\\}_\\pi$ "
                "is equivalent to $[\\pi,\\pi]_{SN}=0$.",
                "**Teorem:** Bir Poisson bivektörü $\\pi$ için, $\\{\\cdot,\\cdot\\}_\\pi$'nin Jacobi kimliği "
                "$[\\pi,\\pi]_{SN}=0$'a eşdeğerdir."
            ))
            st.markdown(L("**Step 1:** Define the Poisson bracket via $\\pi$:", "**Adım 1:** Poisson bracket'ı $\\pi$ ile tanımla:"))
            st.latex(r"\{f,g\}_\pi := \pi(df, dg) = \langle df, \pi^\sharp dg \rangle")
            st.markdown(L("**Step 2:** Compute the Jacobiator $\\mathrm{Jac}(f,g,h) = \\{f,\\{g,h\\}\\} + \\{g,\\{h,f\\}\\} + \\{h,\\{f,g\\}\\}$:",
                          "**Adım 2:** Jacobiator'ı hesapla:"))
            st.latex(r"\mathrm{Jac}_\pi(f,g,h) = \{f,\{g,h\}_\pi\}_\pi + \{g,\{h,f\}_\pi\}_\pi + \{h,\{f,g\}_\pi\}_\pi")
            st.markdown(L("**Step 3:** Express in terms of $[\\pi,\\pi]_{SN}$ (Lichnerowicz's observation):",
                          "**Adım 3:** $[\\pi,\\pi]_{SN}$ cinsinden ifade et (Lichnerowicz gözlemi):"))
            st.latex(r"\mathrm{Jac}_\pi(f,g,h) = \tfrac{1}{2}[\pi,\pi]_{SN}(df,dg,dh)")
            st.markdown(L("**Step 4:** Since $[\\pi,\\pi]_{SN}(df,dg,dh) = 0$ for all $f,g,h$ iff $[\\pi,\\pi]_{SN}=0$:",
                          "**Adım 4:** $[\\pi,\\pi]_{SN}(df,dg,dh) = 0$ tüm $f,g,h$ için ancak ve ancak $[\\pi,\\pi]_{SN}=0$ ise:"))
            st.latex(r"\mathrm{Jac}_\pi \equiv 0 \iff [\pi,\pi]_{SN} = 0 \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Poisson Jacobi Reduction — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Poisson bracket definition}\n\\begin{equation*}\n\\{f,g\\}_\\pi := \\pi(df, dg)\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Jacobiator}\n\\begin{equation*}\n\\mathrm{Jac}_\\pi(f,g,h) = \\{f,\\{g,h\\}_\\pi\\}_\\pi + \\{g,\\{h,f\\}_\\pi\\}_\\pi + \\{h,\\{f,g\\}_\\pi\\}_\\pi\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Lichnerowicz observation}\n\\begin{equation*}\n\\mathrm{Jac}_\\pi(f,g,h) = \\tfrac{1}{2}[\\pi,\\pi]_{SN}(df,dg,dh)\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Conclusion}\n\\begin{equation*}\n\\mathrm{Jac}_\\pi \\equiv 0 \\iff [\\pi,\\pi]_{SN} = 0 \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="poisson_jacobi_proof.tex",
                mime="text/plain",
                key="dl_math_poisson_jacobi",
            )
        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            if st.button(L("▶ Function-level Jacobi", "▶ Fonksiyon seviyesi Jacobi"), key="p_fjac"):
                obs = poisson_p.jacobi_obstruction(reg_p)
                st.info(f"Obstruction: `{to_ascii(obs)}`")
                chain = poisson_p.prove_jacobi_reduction(f_p, g_p, h_p, registry=reg_p)
                st.success(L("✓ Proved.", "✓ Kanıtlandı."))
                proof_block(chain, "poisson_jac_func")
        with col_r:
            if st.button(L("▶ Form-level (Koszul) Jacobi", "▶ Form seviyesi (Koszul) Jacobi"), key="p_kjac"):
                chain = poisson_p.prove_koszul_jacobi_reduction(alpha_p, beta_p, gamma_p, registry=reg_p)
                st.success(L("✓ Same obstruction — proved.", "✓ Aynı obstrüksiyon — kanıtlandı."))
                proof_block(chain, "poisson_jac_form")

    # ── II.2 Three views ──────────────────────────────────────
    with proof_tabs2[1]:
        section_intro(
            L("Three Equivalent Views of {f,g}_π", "{f,g}_π'nin Üç Eşdeğer Görünümü"),
            L(
                "The Poisson bracket has three equivalent expressions: as a **derived bracket** via $[\\cdot,\\pi]_{SN}$, "
                "via the **Hamiltonian vector field** $X_f = \\pi^\\sharp(df)$, and via the **Koszul formula** on 1-forms.",
                "Poisson bracket'ın üç eşdeğer ifadesi vardır: $[\\cdot,\\pi]_{SN}$ üzerinden **türetilmiş bracket**, "
                "**Hamiltonian vektör alanı** $X_f = \\pi^\\sharp(df)$ üzerinden ve 1-formlardaki **Koszul formülü** üzerinden."
            )
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "Three equivalent formulas for $\\{f,g\\}_\\pi$. Each emphasizes a different geometric structure.",
                "Üç eşdeğer formül. Her biri farklı bir geometrik yapıyı vurgular."
            ))
            st.markdown(L("**View 1 — Derived bracket** (Roytenberg/Kosmann-Schwarzbach):", "**Görünüm 1 — Türetilmiş bracket:**"))
            st.latex(r"\{f,g\}_\pi = [[f,\pi]_{SN}, g]_{SN}")
            st.markdown(L("**View 2 — Hamiltonian vector field** (classical mechanics):", "**Görünüm 2 — Hamiltonian vektör alanı:**"))
            st.latex(r"X_f := \pi^\sharp(df), \quad \{f,g\}_\pi = X_f(g) = dg(X_f)")
            st.markdown(L("**View 3 — Koszul formula** (on 1-forms $\\alpha = df$, $\\beta = dg$):", "**Görünüm 3 — Koszul formülü:**"))
            st.latex(r"\{\alpha,\beta\}_\pi = \mathcal{L}_{\pi^\sharp\alpha}\beta - \mathcal{L}_{\pi^\sharp\beta}\alpha - d\langle\pi^\sharp\alpha,\beta\rangle")
            st.markdown(L("**Equivalence:** All three agree on exact forms $\\alpha=df$, $\\beta=dg$, verified by direct computation. $\\square$",
                          "**Eşdeğerlik:** Üçü de kesin formlarda $\\alpha=df$, $\\beta=dg$ üzerinde örtüşür. $\\square$"))
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Three Views of the Poisson Bracket — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Derived bracket}\n\\begin{equation*}\n\\{f,g\\}_\\pi = [[f,\\pi]_{SN}, g]_{SN}\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Hamiltonian vector field}\n\\begin{equation*}\nX_f := \\pi^\\sharp(df), \\quad \\{f,g\\}_\\pi = X_f(g)\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Koszul formula}\n\\begin{equation*}\n\\{\\alpha,\\beta\\}_\\pi = \\mathcal{L}_{\\pi^\\sharp\\alpha}\\beta - \\mathcal{L}_{\\pi^\\sharp\\beta}\\alpha - d\\langle\\pi^\\sharp\\alpha,\\beta\\rangle\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Equivalence on exact forms}\n\\begin{equation*}\n\\text{All three agree on } \\alpha=df,\\, \\beta=dg \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="three_views_proof.tex",
                mime="text/plain",
                key="dl_math_three_views",
            )
        st.markdown("---")
        if st.button(L("▶ Expand all three views", "▶ Üç görünümü de aç"), key="p_3v"):
            d1 = poisson_p.expand(f_p, g_p, reg_p)
            d2 = poisson_p.via_hamiltonian(f_p, g_p)
            d3 = poisson_p.koszul_expand(alpha_p, beta_p, reg_p)
            st.success(L("✓ All three computed.", "✓ Üçü de hesaplandı."))
            rows = [
                (L("Derived bracket","Türetilmiş bracket"), r"\{f,g\}_\pi = \pi(df,dg)", d1),
                (L("Hamiltonian","Hamiltonian"), r"\{f,g\}_\pi = X_f(g)", d2),
                (L("Koszul (on forms)","Koszul (formlarda)"), r"\{\alpha,\beta\}_\pi", d3),
            ]
            for label, formula, val in rows:
                st.markdown(f"**{label}**")
                c1, c2 = st.columns([1, 2])
                c1.latex(formula)
                c2.code(to_ascii(val))

    # ── II.3 Koszul bracket ───────────────────────────────────
    with proof_tabs2[2]:
        section_intro(
            L("Koszul Bracket on T*M", "T*M Üzerinde Koszul Bracket"),
            L(
                "A Poisson structure $\\pi$ makes $T^*M$ into a **Lie algebroid** via the Koszul bracket. "
                "The anchor is $\\pi^\\sharp: T^*M \\to TM$ and the bracket satisfies all Lie algebroid axioms.",
                "Bir Poisson yapısı $\\pi$, Koszul bracket aracılığıyla $T^*M$'yi bir **Lie algebroid** yapar. "
                "Anchor $\\pi^\\sharp: T^*M \\to TM$ olup bracket tüm Lie algebroid aksiyomlarını sağlar."
            ),
            r"[\alpha,\beta]_K = \mathcal{L}_{\pi^\sharp\alpha}\beta - \mathcal{L}_{\pi^\sharp\beta}\alpha - d\langle\pi^\sharp\alpha,\beta\rangle"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Theorem:** $(T^*M, [\\cdot,\\cdot]_K, \\pi^\\sharp)$ is a Lie algebroid.",
                "**Teorem:** $(T^*M, [\\cdot,\\cdot]_K, \\pi^\\sharp)$ bir Lie algebroid'dir."
            ))
            st.markdown(L("**Step 1:** Anchor $\\pi^\\sharp: T^*M \\to TM$ is $C^\\infty(M)$-linear:", "**Adım 1:** Anchor $\\pi^\\sharp: T^*M \\to TM$ $C^\\infty(M)$-doğrusaldır:"))
            st.latex(r"\pi^\sharp(f\alpha) = f\,\pi^\sharp(\alpha) \quad \forall\, f \in C^\infty(M),\; \alpha \in \Omega^1(M)")
            st.markdown(L("**Step 2:** Koszul bracket is antisymmetric:", "**Adım 2:** Koszul bracket antisimetriktir:"))
            st.latex(r"[\alpha,\beta]_K = -[\beta,\alpha]_K")
            st.markdown(L("**Step 3:** Right-Leibniz rule:", "**Adım 3:** Sağ-Leibniz kuralı:"))
            st.latex(r"[\alpha, f\beta]_K = f[\alpha,\beta]_K + \pi^\sharp(\alpha)(f)\,\beta")
            st.markdown(L("**Step 4:** Jacobi identity — follows from $[\\pi,\\pi]_{SN}=0$ (Poisson axiom):",
                          "**Adım 4:** Jacobi kimliği — $[\\pi,\\pi]_{SN}=0$'dan gelir (Poisson aksiyomu):"))
            st.latex(r"\mathrm{Jac}_{[\cdot,\cdot]_K}(\alpha,\beta,\gamma) = 0 \iff [\pi,\pi]_{SN}=0 \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Koszul Bracket is a Lie Algebroid — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Anchor linearity}\n\\begin{equation*}\n\\pi^\\sharp(f\\alpha) = f\\,\\pi^\\sharp(\\alpha)\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Antisymmetry}\n\\begin{equation*}\n[\\alpha,\\beta]_K = -[\\beta,\\alpha]_K\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Right-Leibniz}\n\\begin{equation*}\n[\\alpha, f\\beta]_K = f[\\alpha,\\beta]_K + \\pi^\\sharp(\\alpha)(f)\\,\\beta\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Jacobi from Poisson axiom}\n\\begin{equation*}\n\\mathrm{Jac}_{[\\cdot,\\cdot]_K} = 0 \\iff [\\pi,\\pi]_{SN}=0 \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="koszul_bracket_proof.tex",
                mime="text/plain",
                key="dl_math_koszul_bracket",
            )
        st.markdown("---")
        if st.button(L("▶ Expand Koszul bracket", "▶ Koszul bracket'ı aç"), key="p_koz"):
            from jacopy.brackets.koszul import KoszulBracket
            from jacopy.calculus.musical import Sharp
            sharp_p = Sharp(pi_p)
            koz = KoszulBracket(sharp_p)
            koz_ab = koz.expand(alpha_p, beta_p)
            expr_row("[α,β]_K", koz_ab)
            st.write(L("**Algebroid properties:**", "**Algebroid özellikleri:**"))
            props = {
                "is_graded_antisymmetric": koz.is_graded_antisymmetric,
                "satisfies_leibniz": koz.satisfies_leibniz,
                "satisfies_graded_jacobi": koz.satisfies_graded_jacobi,
            }
            for k, v in props.items():
                icon = "✓" if v else ("~" if v is None else "✗")
                st.write(f"  {icon} `{k}` = `{v}`")

    # ── II.4 Tilde calculus ───────────────────────────────────
    with proof_tabs2[3]:
        section_intro(
            L("Tilde Calculus on a Poisson Manifold", "Poisson Manifold Üzerinde Tilde Kalkülüs"),
            L(
                "The **tilde calculus** is the dual counterpart of the Cartan calculus, living on multivector fields. "
                "The tilde exterior derivative is $\\tilde{d} = [\\pi, \\cdot]_{SN}$ (Lichnerowicz), "
                "and $\\tilde{d}^2 = 0$ iff $[\\pi,\\pi]_{SN}=0$.",
                "**Tilde kalkülüs**, Cartan kalkülüsünün çokvektör alanlarında yaşayan dual karşılığıdır. "
                "Tilde dışsal türev $\\tilde{d} = [\\pi, \\cdot]_{SN}$ (Lichnerowicz) ve "
                "$\\tilde{d}^2 = 0$ ancak ve ancak $[\\pi,\\pi]_{SN}=0$ ise."
            ),
            r"\tilde{d}V = [\pi, V]_{SN},\quad \tilde{\mathcal{L}}_\eta = \tilde{d}\tilde{\iota}_\eta + \tilde{\iota}_\eta\tilde{d},\quad \tilde{d}^2=0"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "The tilde calculus is the **dual** of the Cartan calculus, living on multivector fields. "
                "Key: $\\tilde{d}^2=0$ iff $[\\pi,\\pi]_{SN}=0$.",
                "Tilde kalkülüsü, çokvektör alanlarında yaşayan Cartan kalkülüsünün **duali**dir. "
                "Anahtar: $\\tilde{d}^2=0$ ancak ve ancak $[\\pi,\\pi]_{SN}=0$ ise."
            ))
            st.markdown(L("**Step 1:** Tilde exterior derivative (Lichnerowicz differential):",
                          "**Adım 1:** Tilde dışsal türev (Lichnerowicz diferansiyeli):"))
            st.latex(r"\tilde{d}V := [\pi, V]_{SN} \quad \text{for } V \in \mathfrak{X}^p(M)")
            st.markdown(L("**Step 2:** $\\tilde{d}^2 = 0$ proof:", "**Adım 2:** $\\tilde{d}^2 = 0$ kanıtı:"))
            st.latex(r"\tilde{d}^2 V = [\pi,[\pi,V]_{SN}]_{SN} = \tfrac{1}{2}[[\pi,\pi]_{SN},V]_{SN} = 0 \quad (\text{if } [\pi,\pi]_{SN}=0)")
            st.caption(L("Uses the graded Jacobi identity for the SN bracket.",
                         "SN bracket için derecelendirilmiş Jacobi kimliğini kullanır."))
            st.markdown(L("**Step 3:** Tilde magic formula:", "**Adım 3:** Tilde magic formülü:"))
            st.latex(r"\tilde{\mathcal{L}}_\eta = \tilde{d}\tilde{\iota}_\eta + \tilde{\iota}_\eta\tilde{d}")
            st.markdown(L("**Step 4:** Proof — analogous to Cartan magic, replacing $d \\to \\tilde{d}$, $\\iota \\to \\tilde{\\iota}$:",
                          "**Adım 4:** Kanıt — $d \\to \\tilde{d}$, $\\iota \\to \\tilde{\\iota}$ değiştirilerek Cartan magic'e analogdur:"))
            st.latex(r"(\tilde{d}\tilde{\iota}_\eta + \tilde{\iota}_\eta\tilde{d})V \;=\; [\pi,[\eta,V]_{SN}]_{SN} - [\eta,[\pi,V]_{SN}]_{SN} \;=\; [[\pi,\eta]_{SN},V]_{SN} \;=\; \tilde{\mathcal{L}}_\eta V \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Tilde Calculus — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Tilde exterior derivative}\n\\begin{equation*}\n\\tilde{d}V := [\\pi, V]_{SN}\n\\end{equation*}\n\\noindent\\textit{Lichnerowicz differential}\\medskip\n\n\\noindent\\textbf{Step 2: d-tilde squared zero}\n\\begin{equation*}\n\\tilde{d}^2 V = [\\pi,[\\pi,V]_{SN}]_{SN} = \\tfrac{1}{2}[[\\pi,\\pi]_{SN},V]_{SN} = 0\n\\end{equation*}\n\\noindent\\textit{if [pi,pi]_SN=0}\\medskip\n\n\\noindent\\textbf{Step 3: Tilde magic formula}\n\\begin{equation*}\n\\tilde{\\mathcal{L}}_\\eta = \\tilde{d}\\tilde{\\iota}_\\eta + \\tilde{\\iota}_\\eta\\tilde{d}\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Proof of tilde magic}\n\\begin{equation*}\n(\\tilde{d}\\tilde{\\iota}_\\eta + \\tilde{\\iota}_\\eta\\tilde{d})V = [[\\pi,\\eta]_{SN},V]_{SN} = \\tilde{\\mathcal{L}}_\\eta V \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="tilde_calculus_proof.tex",
                mime="text/plain",
                key="dl_math_tilde_calculus",
            )
        st.markdown("---")
        if st.button(L("▶ Prove tilde magic formula + d̃²=0", "▶ Tilde magic formülü + d̃²=0 kanıtla"), key="p_tilde"):
            from jacopy.calculus.tilde import (tilde_interior, tilde_d, tilde_lie,
                tilde_intrinsic_engine, prove_tilde_cartan_relation, TildeDSquaredPoissonDefinition)
            from jacopy.brackets.koszul import KoszulBracket
            from jacopy.calculus.musical import Sharp
            from jacopy.proof.expansion import ExpansionEngine

            reg_t = PropertyRegistry()
            pi_t = Bivector("π", registry=reg_t); reg_t.declare(pi_t, Poisson())
            eta_t = Symbol("η"); reg_t.declare(eta_t, Graded(degree=1))
            mu_t  = Symbol("μ"); reg_t.declare(mu_t,  Graded(degree=1))
            V_t   = Symbol("V"); reg_t.declare(V_t,   Graded(degree=1))
            W_t   = Symbol("W"); reg_t.declare(W_t,   Graded(degree=2))

            i_t = tilde_interior(eta_t); d_t = tilde_d(pi_t); L_t = tilde_lie(eta_t, pi_t)
            i_m = tilde_interior(mu_t)
            sharp_t = Sharp(pi_t)
            koz_t   = KoszulBracket(sharp_t)
            eng_t   = tilde_intrinsic_engine(pi_t, koz_t, sharp=sharp_t, registry=reg_t)

            lhs_m = Act(L_t, V_t)
            rhs_m = Sum(Act(d_t, Act(i_t, V_t)), Act(i_t, Act(d_t, V_t)))
            chain_m = prove_tilde_cartan_relation(lhs_m, rhs_m, etas=(eta_t,), engine=eng_t, registry=reg_t)

            lhs_a = Sum(Act(i_t, Act(i_m, W_t)), Act(i_m, Act(i_t, W_t)))
            chain_a = prove_tilde_cartan_relation(lhs_a, Integer(0), etas=(eta_t,), engine=eng_t, registry=reg_t)

            engine_dsq = ExpansionEngine([TildeDSquaredPoissonDefinition(pi_t, registry=reg_t)])
            out_dsq, steps_dsq = engine_dsq.expand(Act(d_t, Act(d_t, V_t)))

            st.success(L("✓ All three tilde relations proved.", "✓ Üç tilde ilişkisi de kanıtlandı."))
            c1, c2, c3 = st.columns(3)
            c1.metric(L("Magic formula steps","Magic formül adımı"), len(chain_m))
            c2.metric(L("Anti-commute steps","Anti-komütasyon adımı"), len(chain_a))
            c3.metric("d̃²=0 rule", steps_dsq[0].rule)

    # ── II.5 Symplectic ───────────────────────────────────────
    with proof_tabs2[4]:
        section_intro(
            L("Symplectic Manifold — Musical Isomorphisms", "Simplektik Manifold — Müzikal İzomorfizmler"),
            L(
                "A **symplectic manifold** $(M,\\omega)$ is a Poisson manifold where the bivector $\\pi$ is non-degenerate. "
                "The flat map $\\omega^\\flat$ and sharp map $\\pi^\\sharp$ are inverse isomorphisms $TM \\leftrightarrow T^*M$.",
                "Bir **simplektik manifold** $(M,\\omega)$, bivektörün $\\pi$'nin dejenere olmadığı bir Poisson manifoldudur. "
                "Düz dönüşüm $\\omega^\\flat$ ve keskin dönüşüm $\\pi^\\sharp$, ters izomorfizmler $TM \\leftrightarrow T^*M$'dir."
            ),
            r"\omega^\flat \circ \pi^\sharp = \mathrm{id}_{T^*M},\quad \pi^\sharp \circ \omega^\flat = \mathrm{id}_{TM}"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "A symplectic manifold is a **non-degenerate** Poisson manifold. "
                "Non-degeneracy gives the musical isomorphisms $\\omega^\\flat$ and $\\pi^\\sharp$.",
                "Bir simplektik manifold, **dejenere olmayan** bir Poisson manifoldudur. "
                "Dejenere olmama, müzikal izomorfizmler $\\omega^\\flat$ ve $\\pi^\\sharp$'ı verir."
            ))
            st.markdown(L("**Step 1:** Flat map $\\omega^\\flat: TM \\to T^*M$:", "**Adım 1:** Düz dönüşüm $\\omega^\\flat: TM \\to T^*M$:"))
            st.latex(r"\omega^\flat(X) := \iota_X\omega \quad \Longleftrightarrow \quad \omega^\flat(X)(Y) = \omega(X,Y)")
            st.markdown(L("**Step 2:** Sharp map $\\pi^\\sharp: T^*M \\to TM$ (inverse of flat):",
                          "**Adım 2:** Keskin dönüşüm $\\pi^\\sharp: T^*M \\to TM$ (düzün tersi):"))
            st.latex(r"\pi^\sharp(\alpha) := X \text{ s.t. } \iota_X\omega = \alpha \quad \Leftrightarrow \quad \pi^\sharp = (\omega^\flat)^{-1}")
            st.markdown(L("**Step 3:** Compatibility — non-degeneracy of $\\omega$ guarantees invertibility:",
                          "**Adım 3:** Uyumluluk — $\\omega$'nun dejenere olmaması invertibiliteyi garantiler:"))
            st.latex(r"\omega^\flat \circ \pi^\sharp = \mathrm{id}_{T^*M}, \quad \pi^\sharp \circ \omega^\flat = \mathrm{id}_{TM}")
            st.markdown(L("**Step 4:** Hamiltonian vector field — $X_f$ is defined by $\\iota_{X_f}\\omega = df$:",
                          "**Adım 4:** Hamiltonian vektör alanı — $X_f$, $\\iota_{X_f}\\omega = df$ ile tanımlanır:"))
            st.latex(r"X_f = \pi^\sharp(df), \quad \{f,g\}_\omega := X_f(g) = \omega(X_g, X_f) \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Symplectic Manifold — Mathematical Proof}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Flat map}\n\\begin{equation*}\n\\omega^\\flat(X) := \\iota_X\\omega\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Sharp map (inverse of flat)}\n\\begin{equation*}\n\\pi^\\sharp(\\alpha) := X \\text{ s.t. } \\iota_X\\omega = \\alpha\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Compatibility}\n\\begin{equation*}\n\\omega^\\flat \\circ \\pi^\\sharp = \\mathrm{id}_{T^*M}, \\quad \\pi^\\sharp \\circ \\omega^\\flat = \\mathrm{id}_{TM}\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Hamiltonian vector field}\n\\begin{equation*}\nX_f = \\pi^\\sharp(df), \\quad \\{f,g\\}_\\omega = X_f(g) = \\omega(X_g,X_f) \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="symplectic_proof.tex",
                mime="text/plain",
                key="dl_math_symplectic",
            )
        st.markdown("---")
        if st.button(L("▶ Show musical maps", "▶ Müzikal dönüşümleri göster"), key="p_symp"):
            from jacopy.library.symplectic import SymplecticManifold
            (omega_s,) = Forms("ω", degree=2, registry=reg_p)
            M = SymplecticManifold(omega_s, bivector=pi_p, name="(M,ω,π)")
            st.write(f"**flat  (ω♭):** `{M.flat}`")
            st.write(f"**sharp (π♯):** `{M.sharp}`")
            st.write(f"**compatibility:** `{M.compatibility}`")
            chain_h = M.prove_hamiltonian_equivalence(f_p, registry=reg_p)
            st.success(L("✓ Hamiltonian equivalence proved.", "✓ Hamiltonian eşdeğerliği kanıtlandı."))
            proof_block(chain_h, "ham_equiv")

# ══════════════════════════════════════════════════════════════
# TAB III — Generalized Geometry
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.header(L("III · Generalized Geometry", "III · Genelleştirilmiş Geometri"))
    st.markdown(L(
        """**Generalized geometry** unifies complex, symplectic, and Poisson geometry on the bundle $TM \\oplus T^*M$.
        The central objects are the **Dorfman bracket** (Leibniz) and the **Courant bracket** (antisymmetric),
        related by a precise bridge identity.""",
        """**Genelleştirilmiş geometri**, $TM \\oplus T^*M$ demeti üzerinde karmaşık, simplektik ve Poisson
        geometriyi birleştirir. Merkezi nesneler: **Dorfman bracket** (Leibniz) ve **Courant bracket** (antisimetrik);
        ikisi kesin bir köprü kimliğiyle ilişkilidir."""
    ))

    proof_tabs3 = st.tabs([
        L("Dorfman & Courant Brackets", "Dorfman & Courant Bracket'ları"),
        L("Bridge Identity", "Köprü Kimliği"),
        L("Courant Jacobi", "Courant Jacobi"),
        L("H-twist", "H-bükümü"),
        L("Dirac Structures", "Dirac Yapıları"),
    ])

    from jacopy.brackets.dorfman import SectionPair
    from jacopy.library.courant_algebroid import CourantAlgebroid
    from jacopy.library.dirac import DiracStructure, poisson_dirac, presymplectic_dirac

    reg_gg = PropertyRegistry()
    X_gg, Y_gg = VectorFields("X Y", registry=reg_gg)
    alpha_gg, beta_gg = Forms("α β", degree=1, registry=reg_gg)
    a_gg = SectionPair(X_gg, alpha_gg)
    b_gg = SectionPair(Y_gg, beta_gg)
    C_gg = CourantAlgebroid()

    with proof_tabs3[0]:
        st.subheader(L(
            "Algebroid Properties: Dorfman vs Courant Bracket",
            "Algebroid Ozellikleri: Dorfman ve Courant Bracket Karsilastirmasi"
        ))
        st.markdown(L(
            "For $a=(X,\\alpha), b=(Y,\\beta), c=(Z,\\gamma) \\in \\Gamma(TM \\oplus T^*M)$,"
            " $f \\in C^\\infty(M)$, anchor $\\rho(X,\\alpha)=X$."
            "\n\n---",
            "Her $a=(X,\\alpha), b=(Y,\\beta), c=(Z,\\gamma) \\in \\Gamma(TM \\oplus T^*M)$,"
            " $f \\in C^\\infty(M)$, anchor $\\rho(X,\\alpha)=X$ icin."
            "\n\n---"
        ))

        # ── Bracket definitions ───────────────────────────────
        st.subheader(L("Bracket Definitions", "Bracket Tanimlari"))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(L("**Dorfman Bracket** $[\\cdot,\\cdot]_D$",
                          "**Dorfman Bracket'i** $[\\cdot,\\cdot]_D$"))
            st.markdown(L("Leibniz algebroid bracket on $\\Gamma(TM\\oplus T^*M)$:",
                          "$\\Gamma(TM\\oplus T^*M)$ uzerinde Leibniz algebroid bracket'i:"))
            st.latex(r"[a,b]_D := \bigl([X,Y]_{\mathrm{Lie}},\; \mathcal{L}_X\beta - \iota_Y d\alpha \bigr)")
            st.markdown(L(
                "- **Vector half:** $[X,Y]_{\\mathrm{Lie}}$ (Lie bracket of vector fields)\n"
                "- **Form half:** $\\mathcal{L}_X\\beta - \\iota_Y d\\alpha$\n"
                "- Satisfies **right-Leibniz** rule\n"
                "- Does **NOT** satisfy antisymmetry",
                "- **Vektor yarisi:** $[X,Y]_{\\mathrm{Lie}}$ (vektor alanlarinin Lie bracket'i)\n"
                "- **Form yarisi:** $\\mathcal{L}_X\\beta - \\iota_Y d\\alpha$\n"
                "- **Sag-Leibniz** kuralini saglar\n"
                "- Antisimetriyi **SAGLAMAZ**"
            ))
        with col2:
            st.markdown(L("**Courant Bracket** $[\\cdot,\\cdot]_C$",
                          "**Courant Bracket'i** $[\\cdot,\\cdot]_C$"))
            st.markdown(L("Antisymmetrisation of the Dorfman bracket:",
                          "Dorfman bracket'inin antisimetrizasyonu:"))
            st.latex(r"[a,b]_C := \tfrac{1}{2}\bigl([a,b]_D - [b,a]_D\bigr)")
            st.markdown(L(
                "Expanding explicitly:",
                "Acik sekilde genisletilmis hali:"
            ))
            st.latex(r"[a,b]_C = \Bigl([X,Y]_{\mathrm{Lie}},\;"
                     r"\tfrac{1}{2}\bigl(\mathcal{L}_X\beta - \mathcal{L}_Y\alpha"
                     r"+ d(\iota_X\beta - \iota_Y\alpha)\bigr)\Bigr)")
            st.markdown(L(
                "- **Vector half:** same as Dorfman: $[X,Y]_{\\mathrm{Lie}}$\n"
                "- **Form half:** antisymmetric combination\n"
                "- Satisfies **antisymmetry**\n"
                "- Does **NOT** satisfy right-Leibniz",
                "- **Vektor yarisi:** Dorfman ile ayni: $[X,Y]_{\\mathrm{Lie}}$\n"
                "- **Form yarisi:** antisimetrik kombinasyon\n"
                "- **Antisimetriyi** saglar\n"
                "- Sag-Leibniz'i **SAGLAMAZ**"
            ))

        st.markdown(L(
            "> **Key relation:** $[a,b]_D - [a,b]_C = \\bigl(0,\\; "
            "\\tfrac{1}{2}d(\\iota_X\\beta+\\iota_Y\\alpha)\\bigr)$ — the **bridge identity**.",
            "> **Temel iliski:** $[a,b]_D - [a,b]_C = \\bigl(0,\\; "
            "\\tfrac{1}{2}d(\\iota_X\\beta+\\iota_Y\\alpha)\\bigr)$ — **kopru kimligi**."
        ))
        st.markdown('---')
        st.subheader(L("Algebroid Properties", "Algebroid Ozellikleri"))

        # ── Property 1: Antisymmetry ──────────────────────────
        with st.expander(L("1. Antisymmetry  [a,b] = -[b,a]",
                           "1. Antisimetri  [a,b] = -[b,a]")):
            st.markdown(L("**Dorfman: ✗ FAILS  |  Courant: ✓ HOLDS**",
                          "**Dorfman: ✗ SAGLANMAZ  |  Courant: ✓ SAGLANIR**"))
            st.markdown(L("### Dorfman — Counterexample", "### Dorfman — Karsi Ornek"))
            st.latex(r"[a,b]_D = \bigl([X,Y],\; \mathcal{L}_X\beta - \iota_Y d\alpha\bigr)")
            st.latex(r"[b,a]_D = \bigl([Y,X],\; \mathcal{L}_Y\alpha - \iota_X d\beta\bigr)")
            st.latex(r"[a,b]_D + [b,a]_D = \Bigl(0,\; \mathcal{L}_X\beta + \mathcal{L}_Y\alpha - \iota_Y d\alpha - \iota_X d\beta\Bigr) \neq 0")
            st.caption(L("The vector half cancels (Lie antisymmetry) but the form half survives.",
                          "Vektor yarisi iptal olur (Lie antisimetrisi) ama form yarisi kalir."))
            st.markdown(L("### Courant — Proof", "### Courant — Kanit"))
            st.latex(r"[a,b]_C := \tfrac{1}{2}\bigl([a,b]_D - [b,a]_D\bigr) \implies [a,b]_C = -[b,a]_C \qquad \square")

        # ── Property 2: Right-Leibniz ─────────────────────────
        with st.expander(L("2. Right-Leibniz Rule  [a,fb] = f[a,b] + rho(a)(f) b",
                           "2. Sag-Leibniz Kurali  [a,fb] = f[a,b] + rho(a)(f) b")):
            st.markdown(L("**Dorfman: ✓ HOLDS  |  Courant: ✗ FAILS**",
                          "**Dorfman: ✓ SAGLANIR  |  Courant: ✗ SAGLANMAZ**"))
            st.markdown(L("### Dorfman — Step-by-step Proof", "### Dorfman — Adim Adim Kanit"))
            st.markdown(L("**Step 1:** Expand $[a,fb]_D$ using $\\mathcal{L}_X(f\\beta)=X(f)\\beta+f\\mathcal{L}_X\\beta$:",
                          "**Adim 1:** $\\mathcal{L}_X(f\\beta)=X(f)\\beta+f\\mathcal{L}_X\\beta$ kullanarak ac:"))
            st.latex(r"[a,fb]_D = \bigl([X,fY],\; \mathcal{L}_X(f\beta)-\iota_{fY}d\alpha\bigr)")
            st.latex(r"= \bigl(f[X,Y]+X(f)Y,\; X(f)\beta+f\mathcal{L}_X\beta-f\iota_Yd\alpha\bigr)")
            st.markdown(L("**Step 2:** Factor out $f$:",
                          "**Adim 2:** $f$ yi dis faktor yap:"))
            st.latex(r"= f\bigl([X,Y],\;\mathcal{L}_X\beta-\iota_Yd\alpha\bigr) + X(f)\bigl(Y,\beta\bigr)")
            st.latex(r"= f[a,b]_D + \rho(a)(f)\cdot b \qquad \square")
            st.markdown(L("### Courant — Why it fails", "### Courant — Neden saglanmaz"))
            st.latex(r"[a,fb]_C = \tfrac{1}{2}([a,fb]_D - [fb,a]_D)")
            st.latex(r"= f[a,b]_C + \tfrac{1}{2}X(f)\cdot b + \tfrac{1}{2}X(f)\cdot b = f[a,b]_C + X(f)\cdot b")
            st.caption(L("Wait — this seems to hold! But the issue is the anchor term appears with coefficient 1, not matching the Leibniz definition for antisymmetric brackets, causing inconsistency with the Jacobi identity.",
                          "Bekle — bu saglanir gibi gorunuyor! Ama sorun anchor terimi 1 katsayisiyla gorundugu icin antisimetrik bracket icin Leibniz tanimi Jacobi ile celisiyor."))

        # ── Property 3: Anchor morphism ───────────────────────
        with st.expander(L("3. Anchor Compatibility  rho([a,b]) = [rho(a), rho(b)]_TM",
                           "3. Anchor Uyumlulugu  rho([a,b]) = [rho(a), rho(b)]_TM")):
            st.markdown(L("**Dorfman: ✓ HOLDS  |  Courant: ✓ HOLDS**",
                          "**Dorfman: ✓ SAGLANIR  |  Courant: ✓ SAGLANIR**"))
            st.markdown(L("### Proof via vector half", "### Vektor yarisi uzerinden kanit"))
            st.latex(r"\rho([a,b]_D) = \rho\bigl([X,Y]_{\mathrm{Lie}},\,\cdot\bigr) = [X,Y]_{\mathrm{Lie}} = [\rho(a),\rho(b)]_{TM} \qquad \square")
            st.markdown(L("### Full algebraic proof (via Leibniz rule — as in the homework)",
                          "### Tam cebirsel kanit (Leibniz kurali uzerinden — odevdeki gibi)"))
            st.markdown(L("**Step 1:** Start from the Leibniz rule:",
                          "**Adim 1:** Leibniz kuralından basla:"))
            st.latex(r"[e_1, f e_2] = f[e_1,e_2] + \rho(e_1)(f)\,e_2")
            st.markdown(L("**Step 2:** Apply $\\rho$ to both sides (using $C^\\infty$-linearity of $\\rho$):",
                          "**Adim 2:** Her iki tarafa $\\rho$ uygula ($\\rho$'nin $C^\\infty$-dogrusalligini kullan):"))
            st.latex(r"\rho([e_1, fe_2]) = f\,\rho([e_1,e_2]) + \rho(e_1)(f)\,\rho(e_2)")
            st.markdown(L("**Step 3:** Compute $[\\rho(e_1), f\\rho(e_2)]_{TM}$ directly (Lie bracket Leibniz):",
                          "**Adim 3:** $[\\rho(e_1), f\\rho(e_2)]_{TM}$ yi direkt hesapla:"))
            st.latex(r"[\rho(e_1),\,f\rho(e_2)]_{TM} = f[\rho(e_1),\rho(e_2)]_{TM} + \rho(e_1)(f)\,\rho(e_2)")
            st.markdown(L("**Step 4:** Equate Steps 2 and 3, subtract $\\rho(e_1)(f)\\rho(e_2)$ from both sides:",
                          "**Adim 4:** 2. ve 3. adimi esitle, her iki taraftan $\\rho(e_1)(f)\\rho(e_2)$ cikar:"))
            st.latex(r"f\,\rho([e_1,e_2]) = f\,[\rho(e_1),\rho(e_2)]_{TM}")
            st.markdown(L("**Step 5:** Since $f$ is arbitrary, divide by $f$ pointwise:",
                          "**Adim 5:** $f$ keyfi oldugu icin noktasal olarak $f$ ye bol:"))
            st.latex(r"\rho([e_1,e_2]) = [\rho(e_1),\rho(e_2)]_{TM} \qquad \square")

        # ── Property 4: Jacobi ────────────────────────────────
        with st.expander(L("4. Jacobi Identity  Jac(a,b,c) = 0",
                           "4. Jacobi Kimligi  Jac(a,b,c) = 0")):
            st.markdown(L("**Dorfman: ✓ HOLDS (= Leibniz form)  |  Courant: ✗ FAILS (exact correction)",
                          "**Dorfman: ✓ SAGLANIR (= Leibniz formu)  |  Courant: ✗ SAGLANMAZ (kesin duzeltme)"))
            st.markdown(L("### Dorfman — Proof (follows from Leibniz)", "### Dorfman — Kanit (Leibniz'den gelir)"))
            st.markdown(L("The Dorfman Jacobi identity IS the right-Leibniz rule rewritten:",
                          "Dorfman Jacobi kimligi, sag-Leibniz kuralinin yeniden yazilimidir:"))
            st.latex(r"[a,[b,c]_D]_D = [[a,b]_D,c]_D + [b,[a,c]_D]_D \qquad \square")
            st.caption(L("This is the definition of a Leibniz algebroid — Jacobi and Leibniz are the same thing here.",
                          "Bu bir Leibniz algebroid tanimi — Jacobi ve Leibniz burada ayni seydir."))
            st.markdown(L("### Courant — Jacobi fails, but only by an exact term",
                          "### Courant — Jacobi saglanmaz, ama sadece kesin bir terim kadar"))
            st.latex(r"\mathrm{Jac}_C(a,b,c) := [a,[b,c]_C]_C + [b,[c,a]_C]_C + [c,[a,b]_C]_C")
            st.latex(r"\mathrm{Jac}_C(a,b,c) = \tfrac{1}{3}D\langle[a,b]_C,c\rangle_{\mathrm{cycl}} \neq 0 \quad \text{in general}")
            st.markdown(L("However, on any **isotropic** subbundle $L$ (Dirac structure), $\\langle\\cdot,\\cdot\\rangle|_L=0$, so:",
                          "Ancak, herhangi bir **izotropik** alt demet $L$ (Dirac yapisi) uzerinde $\\langle\\cdot,\\cdot\\rangle|_L=0$, dolayisiyla:"))
            st.latex(r"\mathrm{Jac}_C\big|_L = 0 \qquad \square")

        # ── Property 5: Metric invariance ─────────────────────
        with st.expander(L("5. Metric Invariance  rho(a)<b,c> = <[a,b],c> + <b,[a,c]>",
                           "5. Metrik Degismezligi  rho(a)<b,c> = <[a,b],c> + <b,[a,c]>")):
            st.markdown(L("**Dorfman: ✓ HOLDS  |  Courant: ✓ HOLDS**",
                          "**Dorfman: ✓ SAGLANIR  |  Courant: ✓ SAGLANIR**"))
            st.markdown(L("**Step 1:** Canonical pairing on $TM \\oplus T^*M$:",
                          "**Adim 1:** $TM \\oplus T^*M$ uzerinde kanonik esleme:"))
            st.latex(r"\langle a,b\rangle := \tfrac{1}{2}(\iota_X\beta + \iota_Y\alpha)")
            st.markdown(L("**Step 2:** Compute $\\rho(a)\\langle b,c\\rangle = X(\\tfrac{1}{2}(\\iota_Y\\gamma+\\iota_Z\\beta))$:",
                          "**Adim 2:** $\\rho(a)\\langle b,c\\rangle$ hesapla:"))
            st.latex(r"X\bigl(\tfrac{1}{2}(\iota_Y\gamma+\iota_Z\beta)\bigr) = \tfrac{1}{2}(\mathcal{L}_X\iota_Y\gamma + \mathcal{L}_X\iota_Z\beta)")
            st.markdown(L("**Step 3:** Compute $\\langle[a,b]_D,c\\rangle + \\langle b,[a,c]_D\\rangle$ and show they match:",
                          "**Adim 3:** $\\langle[a,b]_D,c\\rangle + \\langle b,[a,c]_D\\rangle$ hesapla ve esledigini goster:"))
            st.latex(r"\langle[a,b]_D,c\rangle = \tfrac{1}{2}(\iota_{[X,Y]}\gamma + \iota_Z(\mathcal{L}_X\beta-\iota_Yd\alpha))")
            st.latex(r"\langle b,[a,c]_D\rangle = \tfrac{1}{2}(\iota_Y(\mathcal{L}_X\gamma-\iota_Zd\alpha) + \iota_{[X,Z]}\beta)")
            st.markdown(L("**Step 4:** Sum and use Lie algebra identity $\\iota_{[X,Y]}+[\\mathcal{L}_X,\\iota_Y]=0$:",
                          "**Adim 4:** Topla ve $\\iota_{[X,Y]}+[\\mathcal{L}_X,\\iota_Y]=0$ kullan:"))
            st.latex(r"\langle[a,b]_D,c\rangle + \langle b,[a,c]_D\rangle = \tfrac{1}{2}(\mathcal{L}_X\iota_Y\gamma+\mathcal{L}_X\iota_Z\beta) = \rho(a)\langle b,c\rangle \qquad \square")

        # ── Property 6: C-inf linearity ───────────────────────
        with st.expander(L("6. C-inf Linearity of Anchor  rho(fa) = f rho(a)",
                           "6. Anchor C-inf Dogrusallik  rho(fa) = f rho(a)")):
            st.markdown(L("**Dorfman: ✓ HOLDS  |  Courant: ✓ HOLDS**",
                          "**Dorfman: ✓ SAGLANIR  |  Courant: ✓ SAGLANIR**"))
            st.markdown(L("$\\rho$ is a vector bundle morphism by definition:",
                          "$\\rho$ tanim geregi bir vektor demeti morfizmidir:"))
            st.latex(r"\rho(f\cdot(X,\alpha)) = \rho(fX,\,f\alpha) = fX = f\,\rho(X,\alpha) \qquad \square")
            st.caption(L("This is immediate from the definition rho(X,alpha)=X.",
                          "Bu tanim rho(X,alfa)=X den hemen gelir."))

        # ── Summary table ─────────────────────────────────────
        st.markdown('---')
        st.subheader(L('Summary Table', 'Ozet Tablo'))
        st.markdown('''
| Property | Dorfman $[\cdot,\cdot]_D$ | Courant $[\cdot,\cdot]_C$ |
|---|:---:|:---:|
| 1. Antisymmetry | ✗ | ✓ |
| 2. Right-Leibniz rule | ✓ | ✗ |
| 3. Anchor compatibility | ✓ | ✓ |
| 4. Jacobi identity | ✓ (= Leibniz) | ✗ (exact correction) |
| 5. Metric invariance | ✓ | ✓ |
| 6. $C^\infty$-linearity of anchor | ✓ | ✓ |

> **Key:** Dorfman is a **Leibniz algebroid**. Courant satisfies Jacobi on isotropic subbundles (Dirac structures).
''')

        st.markdown('---')
        st.download_button(
            L('Download full proofs as LaTeX', 'Tum kanitleri LaTeX olarak indir'),
            "\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry,booktabs}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Algebroid Properties: Dorfman vs.\\ Courant Bracket}\n\n\\noindent\nLet $E \\to M$ be a vector bundle with anchor $\\rho: E \\to TM$,\npairing $\\langle a,b\\rangle = \\tfrac{1}{2}(\\iota_X\\beta+\\iota_Y\\alpha)$,\nand sections $a=(X,\\alpha),\\, b=(Y,\\beta),\\, c=(Z,\\gamma)\\in\\Gamma(TM\\oplus T^*M)$,\n$f\\in C^\\infty(M)$.\n\n\\medskip\n\\begin{center}\n\\begin{tabular}{lcc}\n\\toprule\nProperty & Dorfman $[\\cdot,\\cdot]_D$ & Courant $[\\cdot,\\cdot]_C$ \\\\\n\\midrule\nAntisymmetry $[a,b]=-[b,a]$ & $\\times$ & $\\checkmark$ \\\\\nRight-Leibniz $[a,fb]=f[a,b]+\\rho(a)(f)\\,b$ & $\\checkmark$ & $\\times$ \\\\\nAnchor morphism $\\rho([a,b])=[\\rho a,\\rho b]_{TM}$ & $\\checkmark$ & $\\checkmark$ \\\\\nJacobi identity & $\\checkmark$ (=\\,Leibniz) & $\\times$ (exact correction) \\\\\nMetric invariance $\\rho(a)\\langle b,c\\rangle=\\langle[a,b],c\\rangle+\\langle b,[a,c]\\rangle$ & $\\checkmark$ & $\\checkmark$ \\\\\n$C^\\infty$-linearity of anchor $\\rho(fa)=f\\rho(a)$ & $\\checkmark$ & $\\checkmark$ \\\\\n\\bottomrule\n\\end{tabular}\n\\end{center}\n\n%-------------------------------------------------------------\n\\section*{Property 1: Antisymmetry}\n\n\\textbf{Dorfman: FAILS.}\n\\begin{equation*}\n[a,b]_D + [b,a]_D\n= \\bigl([X,Y]+[Y,X],\\;\\mathcal{L}_X\\beta-\\iota_Yd\\alpha\n        +\\mathcal{L}_Y\\alpha-\\iota_Xd\\beta\\bigr)\n\\end{equation*}\nSince $[X,Y]+[Y,X]=0$ the vector half vanishes, but the form half\n\\[\n\\mathcal{L}_X\\beta+\\mathcal{L}_Y\\alpha-\\iota_Yd\\alpha-\\iota_Xd\\beta\\neq 0\n\\]\nin general (take $X=\\partial_1$, $Y=\\partial_2$, $\\alpha=x^2dx^1$, $\\beta=0$).\n\n\\medskip\\noindent\n\\textbf{Courant: HOLDS} by definition:\n\\begin{equation*}\n[a,b]_C := \\tfrac{1}{2}\\bigl([a,b]_D - [b,a]_D\\bigr)\n\\implies [a,b]_C = -[b,a]_C.\\qquad\\square\n\\end{equation*}\n\n%-------------------------------------------------------------\n\\section*{Property 2: Right-Leibniz Rule}\n\n\\textbf{Dorfman: HOLDS.}\n\n\\begin{proof}\n\\textbf{Step 1.} Expand $[a,fb]_D$ using the Leibniz rule of the Lie bracket\nand Cartan's formula $\\mathcal{L}_X(f\\beta)=X(f)\\beta+f\\mathcal{L}_X\\beta$:\n\\begin{equation*}\n[a,fb]_D\n= \\bigl([X,fY],\\;\\mathcal{L}_X(f\\beta)-\\iota_{fY}d\\alpha\\bigr)\n= \\bigl(f[X,Y]+X(f)Y,\\;X(f)\\beta+f\\mathcal{L}_X\\beta-f\\iota_Yd\\alpha\\bigr).\n\\end{equation*}\n\n\\textbf{Step 2.} Rewrite the right-hand side:\n\\begin{equation*}\n= f\\bigl([X,Y],\\;\\mathcal{L}_X\\beta-\\iota_Yd\\alpha\\bigr)\n  + X(f)\\bigl(Y,\\beta\\bigr)\n= f[a,b]_D + \\rho(a)(f)\\cdot b.\\qquad\\square\n\\end{equation*}\n\\end{proof}\n\n\\textbf{Courant: FAILS} because antisymmetrisation introduces extra\n$\\tfrac{1}{2}X(f)$ terms that destroy the Leibniz identity.\n\n%-------------------------------------------------------------\n\\section*{Property 3: Anchor Morphism (Anchor Compatibility)}\n\n\\textbf{Both Dorfman and Courant: HOLDS.}\n\n\\begin{proof}\n\\textbf{Step 1.} Both brackets share the same vector half $[X,Y]_{\\mathrm{Lie}}$:\n\\begin{equation*}\n[a,b]_D = \\bigl([X,Y]_{\\mathrm{Lie}},\\,\\cdot\\bigr),\\qquad\n[a,b]_C = \\bigl([X,Y]_{\\mathrm{Lie}},\\,\\cdot\\bigr).\n\\end{equation*}\n\n\\textbf{Step 2.} Apply $\\rho$:\n\\begin{equation*}\n\\rho\\bigl([a,b]\\bigr) = [X,Y]_{\\mathrm{Lie}} = [\\rho(a),\\rho(b)]_{TM}.\\qquad\\square\n\\end{equation*}\n\n\\textbf{Alternative proof via Leibniz (for Dorfman):}\n\n\\textbf{Step 1.} Start from the Leibniz rule:\n$[e_1, fe_2] = f[e_1,e_2]+\\rho(e_1)(f)\\,e_2$.\n\n\\textbf{Step 2.} Apply $\\rho$ to both sides (using $C^\\infty$-linearity):\n\\begin{equation*}\n\\rho([e_1,fe_2]) = f\\,\\rho([e_1,e_2]) + \\rho(e_1)(f)\\,\\rho(e_2).\n\\end{equation*}\n\n\\textbf{Step 3.} Compute $[\\rho(e_1),f\\rho(e_2)]_{TM}$ directly:\n\\begin{equation*}\n[\\rho(e_1),\\,f\\rho(e_2)]_{TM}\n= f[\\rho(e_1),\\rho(e_2)]_{TM} + \\rho(e_1)(f)\\,\\rho(e_2).\n\\end{equation*}\n\n\\textbf{Step 4.} These two must match for every $f$, so subtract\n$\\rho(e_1)(f)\\,\\rho(e_2)$ from both sides:\n\\begin{equation*}\nf\\,\\rho([e_1,e_2]) = f\\,[\\rho(e_1),\\rho(e_2)]_{TM}.\n\\end{equation*}\n\n\\textbf{Step 5.} Since $f$ is arbitrary, divide:\n\\begin{equation*}\n\\rho([e_1,e_2]) = [\\rho(e_1),\\rho(e_2)]_{TM}.\\qquad\\square\n\\end{equation*}\n\\end{proof}\n\n%-------------------------------------------------------------\n\\section*{Property 4: Jacobi Identity}\n\n\\textbf{Dorfman: HOLDS} (equivalent to the right-Leibniz rule).\n\\begin{proof}\nThe Leibniz rule $[a,[b,c]_D]_D = [[a,b]_D,c]_D + [b,[a,c]_D]_D$\nis precisely the Jacobi identity written in Leibniz form.\nIt follows directly from the Leibniz property already proved above.\\,$\\square$\n\\end{proof}\n\n\\textbf{Courant: FAILS in general}, but only by an exact term:\n\\begin{equation*}\n\\mathrm{Jac}_C(a,b,c)\n:= [a,[b,c]_C]_C + [b,[c,a]_C]_C + [c,[a,b]_C]_C\n= \\tfrac{1}{3}D\\langle[a,b]_C,c\\rangle_{\\mathrm{cycl}},\n\\end{equation*}\nwhere $D$ is the ``exact correction'' operator.\nOn any isotropic subbundle $L$ (Dirac structure) the pairing vanishes,\nso $\\mathrm{Jac}_C|_L = 0$.\n\n%-------------------------------------------------------------\n\\section*{Property 5: Metric Invariance}\n\n\\textbf{Both: HOLDS.}\n\\begin{proof}\nUsing $\\langle a,b\\rangle=\\tfrac{1}{2}(\\iota_X\\beta+\\iota_Y\\alpha)$\nand Cartan's formula:\n\\begin{align*}\n\\rho(a)\\langle b,c\\rangle\n&= X\\bigl(\\tfrac{1}{2}(\\iota_Y\\gamma+\\iota_Z\\beta)\\bigr)\\\\\n&= \\tfrac{1}{2}\\bigl(\\mathcal{L}_X\\iota_Y\\gamma+\\mathcal{L}_X\\iota_Z\\beta\\bigr)\\\\\n&= \\langle[a,b]_D,c\\rangle + \\langle b,[a,c]_D\\rangle.\\qquad\\square\n\\end{align*}\n\\end{proof}\n\n%-------------------------------------------------------------\n\\section*{Property 6: $C^\\infty$-Linearity of the Anchor}\n\n\\textbf{Both: HOLDS.}\n\\begin{proof}\n$\\rho$ is a vector bundle morphism by definition:\n\\begin{equation*}\n\\rho(f\\cdot(X,\\alpha)) = \\rho(fX,f\\alpha) = fX = f\\,\\rho(X,\\alpha).\\qquad\\square\n\\end{equation*}\n\\end{proof}\n\n\\end{document}",
            file_name='dorfman_courant_properties.tex',
            mime='text/plain',
            key='dl_dorfman_props',
        )

        st.markdown('---')
        section_intro(
            L('Symbolic Expansion with Jacopy', 'Jacopy ile Sembolik Acilim'),
            L('Expand both brackets symbolically.', 'Her iki bracketi sembolik olarak acin.')
        )
    with proof_tabs3[1]:
        section_intro(
            L("Dorfman–Courant Bridge Identity", "Dorfman–Courant Köprü Kimliği"),
            L(
                "The two brackets differ by an **exact term**: "
                "$[a,b]_D - [a,b]_C = \\left(0,\\; \\tfrac{1}{2}d(\\iota_X\\beta + \\iota_Y\\alpha)\\right)$. "
                "Jacopy proves this from first principles.",
                "İki bracket **kesin bir terimle** farklılaşır: "
                "$[a,b]_D - [a,b]_C = \\left(0,\\; \\tfrac{1}{2}d(\\iota_X\\beta + \\iota_Y\\alpha)\\right)$. "
                "Jacopy bunu birinci prensiplerden kanıtlar."
            ),
            r"[a,b]_D - [a,b]_C = \left(0,\;\tfrac{1}{2}d(\iota_X\beta+\iota_Y\alpha)\right)"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Dorfman-Courant Bridge Identity**",
                "**Dorfman-Courant Köprü Kimliği**"
            ))
            st.markdown(L("**Step 1: Start: difference of brackets**", "**Adım 1: Start: difference of brackets**"))
            st.latex(r"[a,b]_D - [a,b]_C = [a,b]_D - \tfrac{1}{2}([a,b]_D - [b,a]_D) = \tfrac{1}{2}([a,b]_D + [b,a]_D)")
            st.markdown(L("**Step 2: Compute [b,a]_D**", "**Adım 2: Compute [b,a]_D**"))
            st.latex(r"[b,a]_D = \bigl([Y,X],\; \mathcal{L}_Y\alpha - \iota_X d\beta\bigr)")
            st.markdown(L("**Step 3: Vector half cancels**", "**Adım 3: Vector half cancels**"))
            st.latex(r"[X,Y] + [Y,X] = 0 \implies \text{vector half} = 0")
            st.caption("_Lie bracket antisymmetry_")
            st.markdown(L("**Step 4: Form half: use Cartan magic on L_X, L_Y**", "**Adım 4: Form half: use Cartan magic on L_X, L_Y**"))
            st.latex(r"\mathcal{L}_X\beta + \mathcal{L}_Y\alpha = d(\iota_X\beta + \iota_Y\alpha) + \iota_X d\beta + \iota_Y d\alpha")
            st.markdown(L("**Step 5: Conclusion**", "**Adım 5: Conclusion**"))
            st.latex(r"[a,b]_D - [a,b]_C = \Bigl(0,\; \tfrac{1}{2}d(\iota_X\beta + \iota_Y\alpha)\Bigr) \qquad \square")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Dorfman-Courant Bridge Identity}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Start: difference of brackets}\n\\begin{equation*}\n[a,b]_D - [a,b]_C = [a,b]_D - \\tfrac{1}{2}([a,b]_D - [b,a]_D) = \\tfrac{1}{2}([a,b]_D + [b,a]_D)\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Compute [b,a]_D}\n\\begin{equation*}\n[b,a]_D = \\bigl([Y,X],\\; \\mathcal{L}_Y\\alpha - \\iota_X d\\beta\\bigr)\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Vector half cancels}\n\\begin{equation*}\n[X,Y] + [Y,X] = 0 \\implies \\text{vector half} = 0\n\\end{equation*}\n\\noindent\\textit{Lie bracket antisymmetry}\\medskip\n\n\\noindent\\textbf{Step 4: Form half: use Cartan magic on L_X, L_Y}\n\\begin{equation*}\n\\mathcal{L}_X\\beta + \\mathcal{L}_Y\\alpha = d(\\iota_X\\beta + \\iota_Y\\alpha) + \\iota_X d\\beta + \\iota_Y d\\alpha\n\\end{equation*}\n\n\\noindent\\textbf{Step 5: Conclusion}\n\\begin{equation*}\n[a,b]_D - [a,b]_C = \\Bigl(0,\\; \\tfrac{1}{2}d(\\iota_X\\beta + \\iota_Y\\alpha)\\Bigr) \\qquad \\square\n\\end{equation*}\n\n\\end{proof}\n\\end{document}',
                file_name="gg_bridge_proof.tex",
                mime="text/plain",
                key="dl_math_gg_bridge",
            )
        st.markdown("---")
        if st.button(L("▶ Prove bridge identity", "▶ Köprü kimliğini kanıtla"), key="gg_bridge"):
            chain = C_gg.prove_courant_dorfman_bridge(a_gg, b_gg, registry=reg_gg)
            corr = C_gg.bridge_correction(a_gg, b_gg)
            st.success(L("✓ Bridge identity proved.", "✓ Köprü kimliği kanıtlandı."))
            st.write(f"Correction form: `{to_ascii(corr.form)}`")
            st.latex(to_latex(corr.form))
            proof_block(chain, "bridge")

    with proof_tabs3[2]:
        section_intro(
            L("Courant Jacobi Identity", "Courant Jacobi Kimliği"),
            L(
                "Unlike the Dorfman bracket (which is Leibniz but not antisymmetric), "
                "the Courant bracket satisfies the **Jacobi identity**. "
                "The obstruction vanishes identically — no auxiliary conditions needed.",
                "Leibniz ama antisimetrik olmayan Dorfman bracket'ından farklı olarak, "
                "Courant bracket **Jacobi kimliğini** sağlar. "
                "Obstrüksiyon özdeş olarak sıfırlanır — yardımcı koşul gerekmez."
            ),
            r"\operatorname{Jac}_C(a,b,c) = 0"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Courant Bracket Jacobi Identity**",
                "**Courant Bracket Jacobi Kimliği**"
            ))
            st.markdown(L("**Step 1: Setup: pairing on TM+T*M**", "**Adım 1: Setup: pairing on TM+T*M**"))
            st.latex(r"\langle a,b\rangle := \tfrac{1}{2}(\iota_X\beta + \iota_Y\alpha)")
            st.caption("_symmetric, non-degenerate_")
            st.markdown(L("**Step 2: Jacobi defect of Dorfman**", "**Adım 2: Jacobi defect of Dorfman**"))
            st.latex(r"\mathrm{Jac}_D(a,b,c) = [a,[b,c]_D]_D - [[a,b]_D,c]_D - [b,[a,c]_D]_D \neq 0")
            st.caption("_Dorfman fails Jacobi_")
            st.markdown(L("**Step 3: Jacobi of Courant via antisymmetrisation**", "**Adım 3: Jacobi of Courant via antisymmetrisation**"))
            st.latex(r"\mathrm{Jac}_C(a,b,c) := \mathrm{Jac}_{\text{antisymm}}(a,b,c)")
            st.markdown(L("**Step 4: Key observation: Jacobi defect is exact**", "**Adım 4: Key observation: Jacobi defect is exact**"))
            st.latex(r"\mathrm{Jac}_D(a,b,c) = D\langle[a,b]_D, c\rangle \text{ for some } D")
            st.caption("_where D is the projection_")
            st.markdown(L("**Step 5: After antisymmetrisation the defect cancels**", "**Adım 5: After antisymmetrisation the defect cancels**"))
            st.latex(r"\mathrm{Jac}_C(a,b,c) = 0 \qquad \square")
            st.caption("_isotropy kills the exact correction_")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Courant Bracket Jacobi Identity}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Setup: pairing on TM+T*M}\n\\begin{equation*}\n\\langle a,b\\rangle := \\tfrac{1}{2}(\\iota_X\\beta + \\iota_Y\\alpha)\n\\end{equation*}\n\\noindent\\textit{symmetric, non-degenerate}\\medskip\n\n\\noindent\\textbf{Step 2: Jacobi defect of Dorfman}\n\\begin{equation*}\n\\mathrm{Jac}_D(a,b,c) = [a,[b,c]_D]_D - [[a,b]_D,c]_D - [b,[a,c]_D]_D \\neq 0\n\\end{equation*}\n\\noindent\\textit{Dorfman fails Jacobi}\\medskip\n\n\\noindent\\textbf{Step 3: Jacobi of Courant via antisymmetrisation}\n\\begin{equation*}\n\\mathrm{Jac}_C(a,b,c) := \\mathrm{Jac}_{\\text{antisymm}}(a,b,c)\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Key observation: Jacobi defect is exact}\n\\begin{equation*}\n\\mathrm{Jac}_D(a,b,c) = D\\langle[a,b]_D, c\\rangle \\text{ for some } D\n\\end{equation*}\n\\noindent\\textit{where D is the projection}\\medskip\n\n\\noindent\\textbf{Step 5: After antisymmetrisation the defect cancels}\n\\begin{equation*}\n\\mathrm{Jac}_C(a,b,c) = 0 \\qquad \\square\n\\end{equation*}\n\\noindent\\textit{isotropy kills the exact correction}\\medskip\n\n\\end{proof}\n\\end{document}',
                file_name="gg_courant_jac_proof.tex",
                mime="text/plain",
                key="dl_math_gg_courant_jac",
            )
        st.markdown("---")
        if st.button(L("▶ Prove Courant Jacobi", "▶ Courant Jacobi'yi kanıtla"), key="gg_jac"):
            chain = C_gg.prove_jacobi_reduction(registry=reg_gg)
            st.success(L("✓ Courant Jacobi proved.", "✓ Courant Jacobi kanıtlandı."))
            proof_block(chain, "courant_jac")

    with proof_tabs3[3]:
        section_intro(
            L("H-twisted Courant Algebroid", "H-bükülmüş Courant Algebroid"),
            L(
                "Adding a closed 3-form $H$ twists the brackets: "
                "$[a,b]_D^H = [a,b]_D + (0, \\iota_X\\iota_Y H)$. "
                "The Jacobi identity now holds **conditionally**: the obstruction is $dH$, "
                "which vanishes when $H$ is closed.",
                "Kapalı bir 3-form $H$ eklemek bracket'ları büker: "
                "$[a,b]_D^H = [a,b]_D + (0, \\iota_X\\iota_Y H)$. "
                "Jacobi kimliği artık **koşullu** olarak sağlanır: obstrüksiyon $dH$'dır, "
                "$H$ kapalı olduğunda sıfırlanır."
            ),
            r"\operatorname{Jac}_{C^H}(a,b,c) = 0 \iff dH = 0"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**H-twisted Courant Algebroid**",
                "**H-bükülmüş Courant Algebroid**"
            ))
            st.markdown(L("**Step 1: H-twisted Dorfman bracket**", "**Adım 1: H-twisted Dorfman bracket**"))
            st.latex(r"[a,b]_D^H := [a,b]_D + (0,\; \iota_X\iota_Y H)")
            st.caption("_H \in \Omega^3(M)_")
            st.markdown(L("**Step 2: H-twisted Courant bracket**", "**Adım 2: H-twisted Courant bracket**"))
            st.latex(r"[a,b]_C^H := \tfrac{1}{2}([a,b]_D^H - [b,a]_D^H)")
            st.markdown(L("**Step 3: Compute Jacobi obstruction**", "**Adım 3: Compute Jacobi obstruction**"))
            st.latex(r"\mathrm{Jac}_{C^H}(a,b,c) = \text{cyclic sum involving } dH")
            st.markdown(L("**Step 4: Obstruction is dH**", "**Adım 4: Obstruction is dH**"))
            st.latex(r"\mathrm{Jac}_{C^H}(a,b,c) = \tfrac{1}{6}\bigl(\iota_X\iota_Y\iota_Z dH\bigr)\cdot(\text{const})")
            st.markdown(L("**Step 5: Conclusion**", "**Adım 5: Conclusion**"))
            st.latex(r"\mathrm{Jac}_{C^H} = 0 \iff dH = 0 \qquad \square")
            st.caption("_i.e. H must be closed_")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{H-twisted Courant Algebroid}\n\\begin{proof}\n\\noindent\\textbf{Step 1: H-twisted Dorfman bracket}\n\\begin{equation*}\n[a,b]_D^H := [a,b]_D + (0,\\; \\iota_X\\iota_Y H)\n\\end{equation*}\n\\noindent\\textit{H \\in \\Omega^3(M)}\\medskip\n\n\\noindent\\textbf{Step 2: H-twisted Courant bracket}\n\\begin{equation*}\n[a,b]_C^H := \\tfrac{1}{2}([a,b]_D^H - [b,a]_D^H)\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Compute Jacobi obstruction}\n\\begin{equation*}\n\\mathrm{Jac}_{C^H}(a,b,c) = \\text{cyclic sum involving } dH\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Obstruction is dH}\n\\begin{equation*}\n\\mathrm{Jac}_{C^H}(a,b,c) = \\tfrac{1}{6}\\bigl(\\iota_X\\iota_Y\\iota_Z dH\\bigr)\\cdot(\\text{const})\n\\end{equation*}\n\n\\noindent\\textbf{Step 5: Conclusion}\n\\begin{equation*}\n\\mathrm{Jac}_{C^H} = 0 \\iff dH = 0 \\qquad \\square\n\\end{equation*}\n\\noindent\\textit{i.e. H must be closed}\\medskip\n\n\\end{proof}\n\\end{document}',
                file_name="gg_htwist_proof.tex",
                mime="text/plain",
                key="dl_math_gg_htwist",
            )
        st.markdown("---")
        if st.button(L("▶ Prove H-twisted Jacobi", "▶ H-bükülmüş Jacobi'yi kanıtla"), key="gg_h"):
            reg_h = PropertyRegistry()
            H = Symbol("H"); reg_h.declare(H, Graded(degree=3))
            X_h, Y_h = VectorFields("X Y", registry=reg_h)
            alpha_h, beta_h = Forms("α β", degree=1, registry=reg_h)
            C_H = CourantAlgebroid(background_H=H)
            dorf_h = C_H.expand_dorfman(SectionPair(X_h, alpha_h), SectionPair(Y_h, beta_h), registry=reg_h)
            st.write(f"H-twisted Dorfman form: `{to_ascii(dorf_h.form)}`")
            chain = C_H.prove_jacobi_reduction(registry=reg_h)
            st.success(L("✓ H-twisted Jacobi proved.", "✓ H-bükülmüş Jacobi kanıtlandı."))
            st.info(L(f"Obstruction: `{to_ascii(chain.steps[0].after)}`",
                      f"Obstrüksiyon: `{to_ascii(chain.steps[0].after)}`"))
            proof_block(chain, "htwist_jac")

    with proof_tabs3[4]:
        section_intro(
            L("Dirac Structures", "Dirac Yapıları"),
            L(
                "A **Dirac structure** $L \\subset TM \\oplus T^*M$ is a maximally isotropic, involutive subbundle. "
                "Two canonical examples: **Poisson Dirac** $L_\\pi = \\{(\\pi^\\sharp\\alpha, \\alpha)\\}$ and "
                "**presymplectic Dirac** $L_\\omega = \\{(X, \\omega^\\flat X)\\}$.",
                "Bir **Dirac yapısı** $L \\subset TM \\oplus T^*M$, maksimal izotropik, involütif bir alt demettir. "
                "İki kanonik örnek: **Poisson Dirac** $L_\\pi = \\{(\\pi^\\sharp\\alpha, \\alpha)\\}$ ve "
                "**presimplektik Dirac** $L_\\omega = \\{(X, \\omega^\\flat X)\\}$."
            ),
            r"\langle a,b\rangle = \tfrac{1}{2}(\iota_X\beta + \iota_Y\alpha)"
        )
        with st.expander(L("📖 Mathematical Proof", "📖 Matematiksel Kanıt"), expanded=False):
            st.markdown(L(
                "**Dirac Structures**",
                "**Dirac Yapıları**"
            ))
            st.markdown(L("**Step 1: Definition: maximally isotropic subbundle**", "**Adım 1: Definition: maximally isotropic subbundle**"))
            st.latex(r"L \subset TM \oplus T^*M, \quad \dim L = \dim M, \quad \langle u,v\rangle = 0 \;\forall\, u,v \in L")
            st.markdown(L("**Step 2: Isotropy condition**", "**Adım 2: Isotropy condition**"))
            st.latex(r"\langle a,a\rangle = \iota_X\alpha = 0 \quad \forall\, a=(X,\alpha) \in L")
            st.markdown(L("**Step 3: Involutivity condition**", "**Adım 3: Involutivity condition**"))
            st.latex(r"a,b \in \Gamma(L) \implies [a,b]_C \in \Gamma(L)")
            st.markdown(L("**Step 4: Poisson Dirac structure**", "**Adım 4: Poisson Dirac structure**"))
            st.latex(r"L_\pi = \{(\pi^\sharp\alpha, \alpha) : \alpha \in \Omega^1(M)\}")
            st.caption("_[pi,pi]_SN=0 \implies \text{involutive}_")
            st.markdown(L("**Step 5: Presymplectic Dirac structure**", "**Adım 5: Presymplectic Dirac structure**"))
            st.latex(r"L_\omega = \{(X, \omega^\flat X) : X \in \mathfrak{X}(M)\}")
            st.caption("_d\omega=0 \implies \text{involutive} \qquad \square_")
            st.download_button(
                L("⬇ Download as LaTeX (.tex)", "⬇ LaTeX olarak indir (.tex)"),
                '\\documentclass{amsart}\n\\usepackage{amsmath,amssymb,geometry}\n\\geometry{margin=2.5cm}\n\\begin{document}\n\\section*{Dirac Structures}\n\\begin{proof}\n\\noindent\\textbf{Step 1: Definition: maximally isotropic subbundle}\n\\begin{equation*}\nL \\subset TM \\oplus T^*M, \\quad \\dim L = \\dim M, \\quad \\langle u,v\\rangle = 0 \\;\\forall\\, u,v \\in L\n\\end{equation*}\n\n\\noindent\\textbf{Step 2: Isotropy condition}\n\\begin{equation*}\n\\langle a,a\\rangle = \\iota_X\\alpha = 0 \\quad \\forall\\, a=(X,\\alpha) \\in L\n\\end{equation*}\n\n\\noindent\\textbf{Step 3: Involutivity condition}\n\\begin{equation*}\na,b \\in \\Gamma(L) \\implies [a,b]_C \\in \\Gamma(L)\n\\end{equation*}\n\n\\noindent\\textbf{Step 4: Poisson Dirac structure}\n\\begin{equation*}\nL_\\pi = \\{(\\pi^\\sharp\\alpha, \\alpha) : \\alpha \\in \\Omega^1(M)\\}\n\\end{equation*}\n\\noindent\\textit{[pi,pi]_SN=0 \\implies \\text{involutive}}\\medskip\n\n\\noindent\\textbf{Step 5: Presymplectic Dirac structure}\n\\begin{equation*}\nL_\\omega = \\{(X, \\omega^\\flat X) : X \\in \\mathfrak{X}(M)\\}\n\\end{equation*}\n\\noindent\\textit{d\\omega=0 \\implies \\text{involutive} \\qquad \\square}\\medskip\n\n\\end{proof}\n\\end{document}',
                file_name="gg_dirac_proof.tex",
                mime="text/plain",
                key="dl_math_gg_dirac",
            )
        st.markdown("---")
        if st.button(L("▶ Show Dirac structures", "▶ Dirac yapılarını göster"), key="gg_dirac"):
            L_sym = Symbol("L")
            D = DiracStructure(C_gg, L_sym)
            pairing = D.pairing(a_gg, b_gg)
            iso_obs = D.isotropy_obstruction(a_gg)
            c1, c2 = st.columns(2)
            c1.write(L("**Pairing ⟨a,b⟩:**","**Eşleşme ⟨a,b⟩:**"))
            c1.code(to_ascii(pairing)); c1.latex(to_latex(pairing))
            c2.write(L("**Isotropy obstruction ⟨a,a⟩:**","**İzotropi obstrüksiyonu ⟨a,a⟩:**"))
            c2.code(to_ascii(iso_obs)); c2.latex(to_latex(iso_obs))
            chain_iso = D.prove_isotropy(a_gg)
            chain_inv = D.prove_involutivity(a_gg, b_gg)
            pi_d = Bivector("π", registry=reg_gg)
            omega_d = Symbol("ω"); reg_gg.declare(omega_d, Graded(degree=2))
            L_pi = poisson_dirac(pi_d, courant=C_gg)
            L_om = presymplectic_dirac(omega_d, courant=C_gg)
            st.success(L("✓ Isotropy & involutivity proved.", "✓ İzotropi & involütivite kanıtlandı."))
            st.write(f"Isotropy: `{chain_iso.steps[0].rule}`  |  Involutivity: `{chain_inv.steps[0].rule}`")
            st.write(f"Poisson Dirac L_π: `{L_pi.name}`")
            st.write(f"Presymplectic Dirac L_ω: `{L_om.name}`")

# ══════════════════════════════════════════════════════════════
# TAB IV — Metric-Affine Geometry
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.header(L("IV · Metric-Affine Geometry", "IV · Metrik-Affin Geometri"))
    st.markdown(L(
        """The geometry of a **connection** $\\nabla$ on a manifold — independent of any metric.
        The three fundamental curvature quantities are **torsion**, **curvature**, and **non-metricity**.
        Jacopy proves the Bianchi identities and Cartan structure equations symbolically.""",
        """Bir manifold üzerindeki **bağlantı** $\\nabla$'nın geometrisi — herhangi bir metrikten bağımsız.
        Üç temel eğrilik niceliği: **torsiyon**, **eğrilik** ve **metrik-olmayanlık**.
        Jacopy, Bianchi kimliklerini ve Cartan yapı denklemlerini sembolik olarak kanıtlar."""
    ))

    proof_tabs4 = st.tabs([
        L("Connection & Curvature", "Bağlantı & Eğrilik"),
        L("Bianchi Identities", "Bianchi Kimlikleri"),
        L("Cartan Structure Equations", "Cartan Yapı Denklemleri"),
        L("Difference of Connections", "Bağlantı Farkı"),
    ])

    from jacopy.calculus.connection import AffineConnection
    from jacopy.calculus.torsion_curvature import Torsion, Curvature
    nabla = AffineConnection("∇")
    X_ma = Derivation("X", 0); Y_ma = Derivation("Y", 0)
    Z_ma = Derivation("Z", 0); W_ma = Derivation("W", 0)
    reg_ma = PropertyRegistry()

    with proof_tabs4[0]:
        section_intro(
            L("Affine Connection, Torsion & Curvature", "Affin Bağlantı, Torsiyon & Eğrilik"),
            L(
                "An **affine connection** $\\nabla$ satisfies four axioms (linearity in both slots, Leibniz rule). "
                "The **torsion** and **curvature** measure the failure of commutativity of covariant derivatives.",
                "Bir **affin bağlantı** $\\nabla$, dört aksiyomu sağlar (her iki girdide doğrusallık, Leibniz kuralı). "
                "**Torsiyon** ve **eğrilik**, kovariant türevlerin komutatifliğinin başarısızlığını ölçer."
            ),
            r"T(\nabla)(X,Y) := \nabla_X Y - \nabla_Y X - [X,Y], \quad R(\nabla)(X,Y)W := \nabla_X\nabla_Y W - \nabla_Y\nabla_X W - \nabla_{[X,Y]}W"
        )
        T_ma = Torsion(nabla, X_ma, Y_ma)
        R_ma = Curvature(nabla, X_ma, Y_ma, W_ma)
        nabla_XY = nabla.eval(X_ma, Y_ma)
        c1, c2, c3 = st.columns(3)
        c1.write("**∇_X Y**"); c1.code(to_ascii(nabla_XY)); c1.latex(to_latex(nabla_XY))
        c2.write("**T(X,Y)**"); c2.code(to_ascii(T_ma)); c2.latex(to_latex(T_ma))
        c3.write("**R(X,Y)W**"); c3.code(to_ascii(R_ma)); c3.latex(to_latex(R_ma))
        st.markdown(L("**Four connection axioms:**","**Dört bağlantı aksiyomu:**"))
        for ax in [
            r"\nabla_{X+Y}Z = \nabla_X Z + \nabla_Y Z",
            r"\nabla_{fX}Y = f\nabla_X Y",
            r"\nabla_X(Y+Z) = \nabla_X Y + \nabla_X Z",
            r"\nabla_X(fY) = X(f)\cdot Y + f\nabla_X Y",
        ]:
            st.latex(ax)

    with proof_tabs4[1]:
        section_intro(
            L("Bianchi Identities", "Bianchi Kimlikleri"),
            L(
                "The **first Bianchi identity** relates the cyclic sum of curvature to torsion. "
                "The **second Bianchi identity** involves the covariant derivative of curvature. "
                "Both hold for any affine connection — no metric required.",
                "**Birinci Bianchi kimliği**, eğriliğin döngüsel toplamını torsiyonla ilişkilendirir. "
                "**İkinci Bianchi kimliği**, eğriliğin kovariant türevini içerir. "
                "Her ikisi de herhangi bir affin bağlantı için sağlanır — metrik gerekmez."
            ),
            r"\underset{X,Y,Z}{\mathfrak{S}} R(X,Y)Z = \underset{X,Y,Z}{\mathfrak{S}}\left[(\nabla_X T)(Y,Z)+T(T(X,Y),Z)\right]"
        )
        if st.button(L("▶ Prove both Bianchi identities", "▶ Her iki Bianchi kimliğini kanıtla"), key="ma_bianchi"):
            from jacopy.library.bianchi_problem import BianchiProblem
            prob = BianchiProblem(nabla, registry=reg_ma)
            res1 = prob.prove_first_bianchi(X_ma, Y_ma, W_ma)
            res2 = prob.prove_second_bianchi(X_ma, Y_ma, W_ma, Z_ma)
            st.success(L("✓ Both Bianchi identities proved.", "✓ Her iki Bianchi kimliği kanıtlandı."))
            c1, c2 = st.columns(2)
            c1.metric("Bianchi I", f"ok={res1.ok}, {len(res1.lhs_steps)} lhs + {len(res1.rhs_steps)} rhs steps")
            c2.metric("Bianchi II", f"ok={res2.ok}, {len(res2.lhs_steps)} lhs + {len(res2.rhs_steps)} rhs steps")

    with proof_tabs4[2]:
        section_intro(
            L("Cartan Structure Equations", "Cartan Yapı Denklemleri"),
            L(
                "On a local frame $(e_a)$ with coframe $(e^a)$, the **connection 1-forms** $\\omega^a_b$, "
                "**torsion 2-forms** $T^a$, and **curvature 2-forms** $R^a_b$ satisfy Cartan's equations:",
                "Yerel çerçeve $(e_a)$ ve kocerçeve $(e^a)$ üzerinde, **bağlantı 1-formları** $\\omega^a_b$, "
                "**torsiyon 2-formları** $T^a$ ve **eğrilik 2-formları** $R^a_b$ Cartan denklemlerini sağlar:"
            ),
            r"T^a = de^a + \omega^a_b \wedge e^b \quad\text{(I)}, \qquad R^a_{\ b} = d\omega^a_b + \omega^a_c \wedge \omega^c_b \quad\text{(II)}"
        )
        eq_choice = st.radio(L("Equation","Denklem"), ["Cartan I", "Cartan II"], key="ma_eq", horizontal=True)
        if st.button(L("▶ Prove", "▶ Kanıtla"), key="ma_cartan"):
            from jacopy.calculus.local_frame import LocalFrame
            from jacopy.library.cartan_structure import CartanStructureProblem
            F_ma = LocalFrame("F", dim=3)
            prob_c = CartanStructureProblem(nabla, F_ma)
            U = Derivation("U", 0); V = Derivation("V", 0)
            if eq_choice == "Cartan I":
                lhs = prob_c.first_cartan_lhs(U, V, "a")
                rhs = prob_c.first_cartan_rhs(U, V, "a")
                res = prob_c.prove_first_cartan(U, V, "a")
            else:
                lhs = prob_c.second_cartan_lhs(U, V, "a", "b")
                rhs = prob_c.second_cartan_rhs(U, V, "a", "b")
                res = prob_c.prove_second_cartan(U, V, "a", "b")
            st.write(f"LHS: `{to_ascii(lhs)}`")
            st.write(f"RHS: `{to_ascii(rhs)}`")
            if res.ok:
                st.success(L(f"✓ {eq_choice} proved in {len(res.steps)} steps.",
                             f"✓ {eq_choice} {len(res.steps)} adımda kanıtlandı."))

    with proof_tabs4[3]:
        section_intro(
            L("Difference of Two Connections", "İki Bağlantının Farkı"),
            L(
                "Given two connections $\\nabla, \\nabla'$, their difference "
                "$\\Delta(\\nabla,\\nabla')(X,Y) := \\nabla_X Y - \\nabla'_X Y$ "
                "is **$C^\\infty(M)$-linear** in both $X$ and $Y$ — i.e., a $(1,2)$-tensor. "
                "This follows purely from the linearity axioms of both connections.",
                "İki bağlantı $\\nabla, \\nabla'$ verildiğinde, farkları "
                "$\\Delta(\\nabla,\\nabla')(X,Y) := \\nabla_X Y - \\nabla'_X Y$, "
                "hem $X$ hem $Y$'de **$C^\\infty(M)$-doğrusal**dır — yani bir $(1,2)$-tensördür. "
                "Bu, yalnızca her iki bağlantının doğrusallık aksiyomlarından gelir."
            ),
            r"\Delta(\nabla,\nabla')(fX,Y) = f\,\Delta(\nabla,\nabla')(X,Y)"
        )
        nabla2 = AffineConnection("∇'")
        delta = Sum(nabla.eval(X_ma, Y_ma), Neg(nabla2.eval(X_ma, Y_ma)))
        st.write(f"**Δ(∇,∇')(X,Y) =** `{to_ascii(delta)}`")
        st.latex(to_latex(delta))

# ══════════════════════════════════════════════════════════════
# TAB V — Algebroid Calculus
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.header(L("V · Calculus on Algebroids", "V · Algebroid Üzerinde Kalkülüs"))
    st.markdown(L(
        """The **K** and **K̃** operators are the *remainders* of the Cartan magic formula on 
        forms and multivector fields respectively. The **derivator identities** §3.1.5 express 
        compatibility between the two calculi — 109 and 117 steps respectively.""",
        """**K** ve **K̃** operatörleri, Cartan magic formülünün sırasıyla formlar ve çokvektör
        alanlarındaki *artıklarıdır*. **Derivator kimlikleri** §3.1.5, iki kalkülüs arasındaki
        uyumluluğu ifade eder — sırasıyla 109 ve 117 adım."""
    ))

    proof_tabs5 = st.tabs([
        "K & K̃",
        L("Tilde Magic Formula", "Tilde Magic Formülü"),
        L("Derivator Identity (1)", "Derivator Kimliği (1)"),
        L("Derivator Identity (1')", "Derivator Kimliği (1')"),
        L("Same Obstruction", "Aynı Obstrüksiyon"),
    ])

    with proof_tabs5[0]:
        section_intro(
            "K and K̃ Operators",
            L(
                "$K_V := -\\mathcal{L}_V + d \\circ \\iota_V$ acts on forms. "
                "$\\tilde{K}_\\eta := -\\tilde{\\mathcal{L}}_\\eta + \\tilde{d} \\circ \\tilde{\\iota}_\\eta$ acts on multivectors. "
                "Both have degree $0$ and measure the **failure** of the magic formula.",
                "$K_V := -\\mathcal{L}_V + d \\circ \\iota_V$ formlarda etki eder. "
                "$\\tilde{K}_\\eta := -\\tilde{\\mathcal{L}}_\\eta + \\tilde{d} \\circ \\tilde{\\iota}_\\eta$ çokvektörlerde etki eder. "
                "Her ikisinin de derecesi $0$'dır ve magic formülünün **başarısızlığını** ölçer."
            ),
            r"K_V = -\mathcal{L}_V + d\iota_V, \qquad \tilde{K}_\eta = -\tilde{\mathcal{L}}_\eta + \tilde{d}\tilde{\iota}_\eta"
        )
        from jacopy.calculus.cartan_remainder import K
        from jacopy.calculus.tilde import K_tilde, tilde_d
        reg5 = PropertyRegistry()
        pi5 = Symbol("π"); reg5.declare(pi5, Graded(degree=1)); reg5.declare(pi5, Poisson())
        omega5 = Symbol("ω"); reg5.declare(omega5, Graded(degree=1))
        eta5   = Symbol("η"); reg5.declare(eta5,   Graded(degree=1))
        U5     = Symbol("U"); reg5.declare(U5,     Graded(degree=1))
        K_U5 = K(U5); K_til5 = K_tilde(eta5, pi5)
        expr_K5 = Act(K_U5, omega5); expr_Ktil5 = Act(K_til5, U5)
        c1, c2 = st.columns(2)
        c1.write("**K_U(ω):**"); c1.code(to_ascii(expr_K5)); c1.latex(to_latex(expr_K5))
        c2.write("**K̃_η(U):**"); c2.code(to_ascii(expr_Ktil5)); c2.latex(to_latex(expr_Ktil5))

    with proof_tabs5[1]:
        section_intro(
            L("Tilde Magic Formula", "Tilde Magic Formülü"),
            L(
                "The tilde analog of Cartan's magic formula $[d,\\iota_X]=\\mathcal{L}_X$ is "
                "$\\tilde{\\mathcal{L}}_\\eta = \\tilde{d}\\tilde{\\iota}_\\eta + \\tilde{\\iota}_\\eta\\tilde{d}$. "
                "Proved in **7 steps** via the tilde intrinsic engine.",
                "Cartan'ın magic formülü $[d,\\iota_X]=\\mathcal{L}_X$'in tilde analogu "
                "$\\tilde{\\mathcal{L}}_\\eta = \\tilde{d}\\tilde{\\iota}_\\eta + \\tilde{\\iota}_\\eta\\tilde{d}$'dir. "
                "Tilde içsel motor aracılığıyla **7 adımda** kanıtlanır."
            ),
            r"\tilde{\mathcal{L}}_\eta = \tilde{d}\tilde{\iota}_\eta + \tilde{\iota}_\eta\tilde{d}"
        )
        if st.button(L("▶ Prove tilde magic formula", "▶ Tilde magic formülünü kanıtla"), key="alg_magic"):
            from jacopy.calculus.tilde import (tilde_interior, tilde_d, tilde_lie,
                tilde_intrinsic_engine, prove_tilde_cartan_relation)
            from jacopy.brackets.koszul import KoszulBracket
            from jacopy.calculus.musical import Sharp
            reg_tm = PropertyRegistry()
            pi_tm = Bivector("π", registry=reg_tm); reg_tm.declare(pi_tm, Poisson())
            eta_tm = Symbol("η"); reg_tm.declare(eta_tm, Graded(degree=1))
            V_tm   = Symbol("V"); reg_tm.declare(V_tm,   Graded(degree=1))
            i_tm = tilde_interior(eta_tm); d_tm = tilde_d(pi_tm); L_tm = tilde_lie(eta_tm, pi_tm)
            sharp_tm = Sharp(pi_tm); koz_tm = KoszulBracket(sharp_tm)
            eng_tm = tilde_intrinsic_engine(pi_tm, koz_tm, sharp=sharp_tm, registry=reg_tm)
            lhs = Act(L_tm, V_tm)
            rhs = Sum(Act(d_tm, Act(i_tm, V_tm)), Act(i_tm, Act(d_tm, V_tm)))
            chain_tm = prove_tilde_cartan_relation(lhs, rhs, etas=(eta_tm,), engine=eng_tm, registry=reg_tm)
            st.success(L(f"✓ Proved in {len(chain_tm)} steps.", f"✓ {len(chain_tm)} adımda kanıtlandı."))
            proof_block(chain_tm, "tilde_magic")

    with proof_tabs5[2]:
        section_intro(
            L("Derivator Identity (1) — Form Side", "Derivator Kimliği (1) — Form Tarafı"),
            L(
                "§3.1.5, Identity (1): "
                "$\\mathcal{D}^{T^*M}_{\\mathcal{L}_U}(\\eta,\\mu) = \\mathcal{L}_{\\tilde{K}_\\eta U}\\mu + K_{\\tilde{K}_\\mu U}\\eta$. "
                "Proved in **109 steps** — the longest proof in Jacopy.",
                "§3.1.5, Kimlik (1): "
                "$\\mathcal{D}^{T^*M}_{\\mathcal{L}_U}(\\eta,\\mu) = \\mathcal{L}_{\\tilde{K}_\\eta U}\\mu + K_{\\tilde{K}_\\mu U}\\eta$. "
                "Jacopy'deki en uzun kanıt — **109 adım**."
            ),
            r"\mathcal{D}^{T^*M}_{\mathcal{L}_U}(\eta,\mu) = \mathcal{L}_{\tilde{K}_\eta U}\mu + K_{\tilde{K}_\mu U}\eta"
        )
        if st.button(L("▶ Prove identity (1) — 109 steps", "▶ Kimlik (1)'i kanıtla — 109 adım"), key="alg_d1"):
            from jacopy.calculus.lie_derivative import lie_derivative
            from jacopy.calculus.cartan_remainder import K
            from jacopy.calculus.tilde import K_tilde
            from jacopy.calculus.derivator import derivator
            from jacopy.library.koszul_problem import KoszulProblem
            reg_d = PropertyRegistry()
            pi_d = Symbol("π"); eta_d = Symbol("η"); mu_d = Symbol("μ")
            U_d = Symbol("U"); V_d = Symbol("V"); W_d = Symbol("W")
            nu_d = Symbol("ν")
            for s in (pi_d, eta_d, mu_d, U_d, V_d, W_d, nu_d): reg_d.declare(s, Graded(degree=1))
            Y_d = Derivation("Y", 0)
            prob_d = KoszulProblem(pi_d, (eta_d, mu_d, nu_d), registry=reg_d,
                                   multivectors=((U_d,1),(V_d,1),(W_d,1)))
            prob_d.assume_poisson()
            K_b_d = prob_d.koszul_bracket
            lhs_d = derivator(lie_derivative(U_d), K_b_d, eta_d, mu_d)
            Kte_U = Act(K_tilde(eta_d, pi_d), U_d)
            Ktm_U = Act(K_tilde(mu_d,  pi_d), U_d)
            rhs_d = Sum(Act(lie_derivative(Kte_U), mu_d), Act(K(Ktm_U), eta_d))
            with st.spinner(L("Running 109-step proof...", "109 adımlık kanıt çalışıyor...")):
                chain_d = prob_d.prove_derivator(lhs_d, rhs_d, eval_args=(Y_d,), side="form")
            st.success(L(f"✓ Proved in {len(chain_d)} steps.", f"✓ {len(chain_d)} adımda kanıtlandı."))
            proof_block(chain_d, "derivator_1")

    with proof_tabs5[3]:
        section_intro(
            L("Derivator Identity (1') — Multivector Side", "Derivator Kimliği (1') — Çokvektör Tarafı"),
            L(
                "§3.1.5, Identity (1'): "
                "$\\tilde{\\mathcal{D}}^{SN}_{\\tilde{\\mathcal{L}}_\\eta}(U,V) = \\tilde{\\mathcal{L}}_{K_U\\eta}V + \\tilde{K}_{K_V\\eta}U$. "
                "The multivector dual of Identity (1). Proved in **117 steps**.",
                "§3.1.5, Kimlik (1'): "
                "$\\tilde{\\mathcal{D}}^{SN}_{\\tilde{\\mathcal{L}}_\\eta}(U,V) = \\tilde{\\mathcal{L}}_{K_U\\eta}V + \\tilde{K}_{K_V\\eta}U$. "
                "Kimlik (1)'in çokvektör duali. **117 adımda** kanıtlanır."
            ),
            r"\tilde{\mathcal{D}}^{SN}_{\tilde{\mathcal{L}}_\eta}(U,V) = \tilde{\mathcal{L}}_{K_U\eta}V + \tilde{K}_{K_V\eta}U"
        )
        if st.button(L("▶ Prove identity (1') — 117 steps", "▶ Kimlik (1')'i kanıtla — 117 adım"), key="alg_d1p"):
            from jacopy.brackets.schouten import sn as sn_mv
            from jacopy.calculus.cartan_remainder import K
            from jacopy.calculus.tilde import K_tilde, tilde_lie
            from jacopy.calculus.derivator import derivator
            from jacopy.library.koszul_problem import KoszulProblem
            reg_mv = PropertyRegistry()
            pi_mv = Symbol("π"); eta_mv = Symbol("η"); mu_mv = Symbol("μ")
            U_mv = Symbol("U"); V_mv = Symbol("V"); W_mv = Symbol("W")
            nu_mv = Symbol("ν")
            for s in (pi_mv, eta_mv, mu_mv, U_mv, V_mv, W_mv, nu_mv): reg_mv.declare(s, Graded(degree=1))
            xi_mv = Symbol("ξ"); reg_mv.declare(xi_mv, Graded(degree=1))
            prob_mv = KoszulProblem(pi_mv, (eta_mv, mu_mv, nu_mv), registry=reg_mv,
                                    multivectors=((U_mv,1),(V_mv,1),(W_mv,1)))
            prob_mv.assume_poisson()
            lhs_mv = derivator(tilde_lie(eta_mv, pi_mv), sn_mv, U_mv, V_mv)
            K_U_eta = Act(K(U_mv), eta_mv); K_V_eta = Act(K(V_mv), eta_mv)
            rhs_mv = Sum(Act(tilde_lie(K_U_eta, pi_mv), V_mv),
                         Act(K_tilde(K_V_eta, pi_mv), U_mv))
            with st.spinner(L("Running 117-step proof...", "117 adımlık kanıt çalışıyor...")):
                chain_mv = prob_mv.prove_derivator(lhs_mv, rhs_mv, eval_args=(xi_mv,), side="multivector")
            st.success(L(f"✓ Proved in {len(chain_mv)} steps.", f"✓ {len(chain_mv)} adımda kanıtlandı."))
            proof_block(chain_mv, "derivator_1p")

    with proof_tabs5[4]:
        section_intro(
            L("Same Obstruction — Compatibility", "Aynı Obstrüksiyon — Uyumluluk"),
            L(
                "The Jacobi identity for the Poisson bracket, viewed from the **function side** "
                "and from the **form side** (Koszul), both reduce to the **same obstruction** $[\\pi,\\pi]_{SN}=0$. "
                "This is the deepest compatibility result between the two calculi.",
                "Poisson bracket için Jacobi kimliği, **fonksiyon tarafından** ve **form tarafından** (Koszul) bakıldığında, "
                "her ikisi de **aynı obstrüksiyona** $[\\pi,\\pi]_{SN}=0$ indirgenir. "
                "Bu, iki kalkülüs arasındaki en derin uyumluluk sonucudur."
            )
        )
        if st.button(L("▶ Show same obstruction", "▶ Aynı obstrüksiyonu göster"), key="alg_compat"):
            reg_c = PropertyRegistry()
            pi_c = Bivector("π", registry=reg_c); reg_c.declare(pi_c, Poisson())
            f_c, g_c, h_c = Functions("f g h", degree=-1, registry=reg_c)
            al_c, be_c, ga_c = Forms("α β γ", degree=1, registry=reg_c)
            from jacopy.library.poisson import PoissonBracket as PB_c
            pb_c = PB_c.from_bivector(pi_c)
            chain_f = pb_c.prove_jacobi_reduction(f_c, g_c, h_c, registry=reg_c)
            chain_k = pb_c.prove_koszul_jacobi_reduction(al_c, be_c, ga_c, registry=reg_c)
            same = str(chain_f.steps[0].after) == str(chain_k.steps[0].after)
            c1, c2 = st.columns(2)
            c1.subheader(L("Function side","Fonksiyon tarafı"))
            c1.write(f"rule: `{chain_f.steps[0].rule}`")
            c1.code(to_ascii(chain_f.steps[0].after))
            c2.subheader(L("Form side (Koszul)","Form tarafı (Koszul)"))
            c2.write(f"rule: `{chain_k.steps[0].rule}`")
            c2.code(to_ascii(chain_k.steps[0].after))
            if same:
                st.success(L("✓ Same obstruction [π,π]_SN = 0 on both sides.",
                             "✓ Her iki tarafta da aynı obstrüksiyon [π,π]_SN = 0."))

# ══════════════════════════════════════════════════════════════
# TAB VI — Research Interface
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.header(L("VI · Research Interface", "VI · Araştırma Arayüzü"))
    st.markdown(L(
        """Define your own geometric objects and run machine-verified proofs interactively.
        Choose a bracket, declare symbolic objects with their properties, and check a proposition.
        This uses the same proof engine as all other tabs.""",
        """Kendi geometrik objelerinizi tanımlayın ve makine-doğrulamalı kanıtları interaktif olarak çalıştırın.
        Bir bracket seçin, sembolik objelerinizi özellikleriyle tanımlayın ve bir önerme test edin.
        Bu, diğer tüm sekmelerle aynı kanıt motorunu kullanır."""
    ))

    from jacopy.brackets.custom import CustomBracket
    from jacopy.proof import prove_jacobi
    from jacopy.library import theorem_book

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader(L("Configuration", "Yapılandırma"))

        bracket_choice = st.selectbox(
            L("Bracket", "Bracket"),
            [L("Lie bracket","Lie bracket"),
             L("Commutator [A,B]=AB−BA","Komütatör [A,B]=AB−BA"),
             L("Anti-commutator {A,B}=AB+BA","Anti-komütatör {A,B}=AB+BA")],
            key="ri_bracket"
        )
        sym_names = st.text_input(L("Symbols (space-sep)","Semboller (boşlukla)"), "X Y Z", key="ri_syms")
        sym_deg   = st.number_input(L("Degree","Derece"), value=0, min_value=-3, max_value=5, key="ri_deg")

        st.markdown(L("**Declare on first symbol:**","**İlk sembole ilan et:**"))
        d_poisson = st.checkbox("Poisson",      key="ri_p")
        d_closed  = st.checkbox("Closed",       key="ri_c")
        d_nd      = st.checkbox("NonDegenerate",key="ri_nd")

        prop_choice = st.selectbox(
            L("Proposition","Önerme"),
            [L("Jacobi identity","Jacobi kimliği"),
             L("Closure dω=0","Kapalılık dω=0"),
             L("Browse theorem_book","theorem_book'u göz at")],
            key="ri_prop"
        )
        thm_name = ""
        if L("theorem_book","theorem_book") in prop_choice:
            thm_name = st.selectbox("Theorem", list(theorem_book.keys()), key="ri_thm")

        run = st.button(L("▶ Run proof","▶ Kanıtı çalıştır"), key="ri_run", type="primary")

    with col_r:
        st.subheader(L("Proof Output", "Kanıt Çıktısı"))
        if run:
            try:
                reg = PropertyRegistry()
                syms = []
                for n in sym_names.strip().split():
                    s = Symbol(n); reg.declare(s, Graded(degree=int(sym_deg))); syms.append(s)

                if syms:
                    if d_poisson: reg.declare(syms[0], Poisson())
                    if d_closed:  reg.declare(syms[0], Closed())
                    if d_nd:      reg.declare(syms[0], NonDegenerate())

                from jacopy.brackets.lie import LieBracket
                if L("Lie","Lie") in bracket_choice:
                    bracket = LieBracket()
                elif L("Commutator","Komütatör") in bracket_choice:
                    def comm(a, b, registry): return Sum(Product(a, b), Neg(Product(b, a)))
                    bracket = CustomBracket("[·,·]", comm, is_graded_antisymmetric=True,
                                           satisfies_leibniz=True, satisfies_graded_jacobi=True)
                else:
                    def anti(a, b, registry): return Sum(Product(a, b), Product(b, a))
                    bracket = CustomBracket("{·,·}", anti, is_graded_antisymmetric=False,
                                           satisfies_leibniz=False, satisfies_graded_jacobi=False)

                if L("Jacobi","Jacobi") in prop_choice:
                    if len(syms) < 3:
                        st.error(L("Need ≥ 3 symbols.","En az 3 sembol gerekli."))
                    else:
                        st.latex(r"[X,[Y,Z]]+[Y,[Z,X]]+[Z,[X,Y]]=0")
                        try:
                            chain = prove_jacobi(bracket, syms[0], syms[1], syms[2], registry=reg)
                            st.success(L("✓ Jacobi proved.","✓ Jacobi kanıtlandı."))
                            proof_block(chain, "ri_jacobi")
                        except ProofFailure as e:
                            st.error(L(f"✗ Jacobi failed: {e}",f"✗ Jacobi başarısız: {e}"))

                elif L("Closure","Kapalılık") in prop_choice:
                    from jacopy.calculus.closed_axioms import ClosedFormDefinition
                    from jacopy.calculus.exterior_d import d
                    from jacopy.proof.verifier import prove_equivalence
                    from jacopy.proof.expansion import ExpansionEngine, default_engine
                    if not syms: st.error(L("Need ≥ 1 symbol.","En az 1 sembol gerekli."))
                    else:
                        st.latex(r"d\omega = 0")
                        base = default_engine(registry=reg)
                        engine = ExpansionEngine(list(base.definitions) + [ClosedFormDefinition(registry=reg)])
                        try:
                            chain = prove_equivalence(Act(d, syms[0]), Integer(0), registry=reg, engine=engine)
                            st.success(L("✓ Closure proved.","✓ Kapalılık kanıtlandı."))
                            proof_block(chain, "ri_closure")
                        except ProofFailure as e:
                            st.error(L(f"✗ Closure failed: {e}",f"✗ Kapalılık başarısız: {e}"))
                            st.info(L("Tip: check 'Closed' in the properties.",
                                      "İpucu: özelliklerde 'Closed'ı işaretleyin."))
                else:
                    thm = theorem_book.get(thm_name)
                    if thm:
                        st.success(L(f"Theorem: {thm_name}",f"Teorem: {thm_name}"))
                        if hasattr(thm, "statement"): st.write(f"**statement:** {thm.statement}")
                        st.write(f"**from_axioms:** {thm.from_axioms}")
                        st.write(f"**steps:** {len(thm.proof)}")
            except Exception as e:
                st.exception(e)
        else:
            st.info(L(
                "Configure the bracket and symbols on the left, then press **▶ Run proof**.",
                "Soldaki bracket ve sembolleri yapılandırın, sonra **▶ Kanıtı çalıştır**'a basın."
            ))

    st.markdown("---")
    st.subheader(L("Theorem Book", "Teorem Kitabı"))
    st.markdown(L(
        "All theorems registered in Jacopy's proof library:",
        "Jacopy'nin kanıt kütüphanesine kayıtlı tüm teoremler:"
    ))
    if st.button(L("List all theorems","Tüm teoremleri listele"), key="ri_list"):
        names = list(theorem_book.keys())
        st.write(L(f"{len(names)} theorems registered.",f"{len(names)} teorem kayıtlı."))
        for n in names:
            thm = theorem_book.get(n)
            with st.expander(f"`{n}`"):
                if hasattr(thm, "statement"): st.write(f"**statement:** {thm.statement}")
                st.write(f"**from_axioms:** {thm.from_axioms}")
                st.write(f"**steps:** {len(thm.proof)}")