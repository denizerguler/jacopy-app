import sys
from pathlib import Path
import streamlit as st

here = Path.cwd().resolve()
for candidate in (here, *here.parents):
    if (candidate / "jacopy" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

st.set_page_config(page_title="Jacopy Proof Gallery", layout="wide", page_icon="∂")

# ── Language ──────────────────────────────────────────────────
lang = st.sidebar.radio("🌐 Language / Dil", ["English", "Türkçe"])
TR = lang == "Türkçe"
def L(en, tr): return tr if TR else en

# ── Core imports ──────────────────────────────────────────────
from jacopy import VectorFields, Forms
from jacopy.core.expr import Symbol, Sum, Product, Neg, Integer
from jacopy.core.properties import Graded, Closed, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ProofFailure
from jacopy.display import chain_to_ascii, chain_to_latex
from jacopy.display import to_ascii, to_latex
from jacopy.algebra.derivation import Derivation, Act
from jacopy.library.declarations import Bivector, Functions

# ── Helpers ───────────────────────────────────────────────────
def show_chain(chain, key="proof"):
    st.success(L("Done.", "Tamamlandı."))
    c1, c2 = st.columns(2)
    c1.metric(L("Steps", "Adım sayısı"), len(chain))
    c2.write(f"**{L('Result','Sonuç')}:** `{chain.steps[-1].after}`")
    st.subheader(L("Step by step", "Adım adım"))
    for i, step in enumerate(chain.steps, 1):
        with st.expander(f"{L('Step','Adım')} {i}: {step.rule}", expanded=True):
            a, b = st.columns(2)
            a.write(L("Before:", "Önce:"))
            a.code(str(step.before), language="text")
            b.write(L("After:", "Sonra:"))
            b.code(str(step.after), language="text")
            if step.justification:
                st.caption(step.justification)
    with st.expander("ASCII"):
        st.code(chain_to_ascii(chain), language="text")
    with st.expander("LaTeX"):
        try:
            ltx = chain_to_latex(chain)
            st.code(ltx, language="latex")
            st.download_button(L("Download LaTeX","LaTeX İndir"), ltx, f"{key}.tex", key=f"dl_{key}")
        except Exception as e:
            st.warning(L("LaTeX export failed.","LaTeX üretilemedi."))

def show_expr(label, expr):
    st.write(f"**{label}**")
    c1, c2 = st.columns(2)
    c1.code(to_ascii(expr), language="text")
    c2.code(to_latex(expr), language="latex")

def err(e):
    st.error(f"{L('Failed','Başarısız')}: {e}")

# ── Title ─────────────────────────────────────────────────────
st.title(L("Jacopy Proof Gallery", "Jacopy İspat Galerisi"))
st.caption(L(
    "Interactive proofs organized by the project specification categories.",
    "Proje spesifikasyonu kategorilerine göre düzenlenmiş interaktif kanıtlar."
))

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    L("Central Code",          "Merkezi Kod"),
    L("Generalized Geometry",  "Genelleştirilmiş Geometri"),
    L("Metric-Affine Geometry","Metrik-Affin Geometri"),
    L("Symplectic & Poisson",  "Simplektik & Poisson"),
    L("Calculus on Algebroids","Algebroid Kalkülüs"),
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Central Code  (spec §8, §9, §10)
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.header(L("Central Code — Basic Objects & Operations", "Merkezi Kod — Temel Nesneler"))
    st.markdown(L(
        "Basic symbolic objects (§8), tangent bundle operations (§9), algebroid objects (§10).",
        "Temel sembolik nesneler (§8), teğet demet işlemleri (§9), algebroid nesneler (§10)."
    ))

    sub = st.radio(L("Section","Bölüm"), [
        L("§8 Basic objects","§8 Temel nesneler"),
        L("§9a Lie bracket & Jacobi","§9a Lie bracket & Jacobi"),
        L("§9f Schouten–Nijenhuis","§9f Schouten–Nijenhuis"),
        L("§9g Cartan relations (all 5)","§9g Cartan ilişkileri (5 adet)"),
        L("§10 Algebroid anchor compatibility","§10 Algebroid anchor uyumluluğu"),
    ], key="cc_sub")

    if L("§8","§8") in sub:
        from jacopy.algorithms.simplify import simplify
        from jacopy.core.wedge import Wedge
        reg = PropertyRegistry()
        f, g = Functions("f g", registry=reg)
        U, V = VectorFields("U V", registry=reg)
        omega, eta = Forms("ω η", degree=1, registry=reg)
        T = Symbol("T"); reg.declare(T, Graded(degree=1))
        pi = Bivector("π", registry=reg)

        st.markdown("### §8d–j Basic objects")
        c1, c2 = st.columns(2)
        with c1:
            st.write(L("**Functions** C∞(M):","**Fonksiyonlar** C∞(M):"))
            st.code(f"f = {to_ascii(f)},  g = {to_ascii(g)}")
            st.write(L("**Vector fields:**","**Vektör alanları:**"))
            st.code(f"U = {to_ascii(U)},  V = {to_ascii(V)}")
            st.write(L("**1-forms:**","**1-formlar:**"))
            st.code(f"ω = {to_ascii(omega)},  η = {to_ascii(eta)}")
        with c2:
            st.write(L("**Bivector (2-vector):**","**Bivektör:**"))
            st.code(f"π = {to_ascii(pi)}")
            st.write(L("**Wedge product ∧:**","**Dış çarpım ∧:**"))
            st.code(to_ascii(Wedge(omega, eta)))
            st.latex(to_latex(Wedge(omega, eta)))
            st.write(L("**f·T:**","**f·T:**"))
            st.code(to_ascii(Product(f, T)))
            st.latex(to_latex(Product(f, T)))

        st.markdown("### §8c Operations: +, −, ·")
        expr = Sum(f, Sum(f, Neg(f)))
        result = simplify(expr, reg)
        c1, c2 = st.columns(2)
        c1.write(L("Input:","Girdi:")); c1.code(to_ascii(expr))
        c2.write(L("Simplified:","Basitleştirilmiş:")); c2.code(to_ascii(result))

        st.markdown("### §8f Action U(f) & §8r Interior ι_U(ω)")
        U_der = Derivation("U", 0)
        iota_X = __import__("jacopy.calculus.interior", fromlist=["interior"]).interior(U_der)
        Uf = Act(U_der, f)
        iota_omega = Act(iota_X, omega)
        c1, c2 = st.columns(2)
        c1.write("U(f):"); c1.code(to_ascii(Uf)); c1.latex(to_latex(Uf))
        c2.write("ι_U(ω):"); c2.code(to_ascii(iota_omega)); c2.latex(to_latex(iota_omega))

    elif L("§9a","§9a") in sub:
        from jacopy.brackets.lie import lie
        from jacopy.proof import prove_jacobi
        st.latex(r"[X,[Y,Z]]+[Y,[Z,X]]+[Z,[X,Y]]=0")
        names = st.text_input(L("Vector fields","Vektör alanları"), "X Y Z")
        if st.button(L("Prove","Kanıtla"), key="cc_lie"):
            try:
                reg = PropertyRegistry()
                syms = VectorFields(names, registry=reg)
                chain = prove_jacobi(lie, syms[0], syms[1], syms[2], registry=reg)
                show_chain(chain, "lie_jacobi")
            except ProofFailure as e: err(e)

    elif L("§9f","§9f") in sub:
        from jacopy.brackets.schouten import sn
        st.latex(r"[X,Y]_{SN},\quad [f,g]_{SN}=0,\quad [X,f]_{SN}=X(f)")
        if st.button(L("Compute","Hesapla"), key="cc_sn"):
            reg = PropertyRegistry()
            X = Symbol("X"); reg.declare(X, Graded(degree=0))
            Y = Symbol("Y"); reg.declare(Y, Graded(degree=0))
            f = Symbol("f"); reg.declare(f, Graded(degree=-1))
            results = {
                "[X,Y]_SN": sn.expand(X, Y, reg),
                "[f,f]_SN": sn.expand(f, f, reg),
                "[X,f]_SN": sn.expand(X, f, reg),
                "[f,X]_SN": sn.expand(f, X, reg),
            }
            st.success(L("Done.","Tamamlandı."))
            for label, val in results.items():
                c1, c2 = st.columns([1,3])
                c1.code(label); c2.code(to_ascii(val))

    elif L("Cartan","Cartan") in sub:
        from jacopy.calculus.cartan import CartanCalculus, RELATIONS
        from jacopy.calculus.exterior_algebra import ExteriorAlgebra
        from jacopy.calculus.exterior_d import d
        from jacopy.calculus.interior import interior
        from jacopy.calculus.lie_derivative import lie_derivative
        from jacopy.brackets.lie import LieBracket
        st.markdown(L(
            "1. `d²=0`  2. `[d,ι_X]=L_X`  3. `[d,L_X]=0`  4. `[L_X,L_Y]=L_{[X,Y]}`  5. `[L_X,ι_Y]=ι_{[X,Y]}`",
            "1. `d²=0`  2. `[d,ι_X]=L_X`  3. `[d,L_X]=0`  4. `[L_X,L_Y]=L_{[X,Y]}`  5. `[L_X,ι_Y]=ι_{[X,Y]}`"
        ))
        if st.button(L("Verify all 5","Tümünü doğrula"), key="cc_cartan"):
            reg = PropertyRegistry()
            f0 = Symbol("f"); reg.declare(f0, Graded(degree=0))
            alg = ExteriorAlgebra((f0,))
            X0 = Derivation("X", 0); Y0 = Derivation("Y", 0)
            cart = CartanCalculus(d=d, lie_derivative=lie_derivative,
                                  interior=interior, vector_bracket=LieBracket())
            results = cart.verify_all(algebra=alg, X=X0, Y=Y0, registry=reg)
            st.success(L("All 5 verified ✅","Tüm 5 ilişki doğrulandı ✅"))
            for name, chain in results.items():
                st.write(f"**{name}** — {len(chain)} {L('step(s)','adım')}")

    else:  # §10 anchor
        from jacopy.calculus.anchor import Anchor
        from jacopy.brackets.lie import LieBracket
        from jacopy.library.lie_algebroid import LieAlgebroid
        st.latex(r"\rho([X,Y]_E)=[\rho(X),\rho(Y)]_{TM}")
        st.markdown(L(
            "Anchor compatibility is a **separate axiom** — not derivable from Lie bracket axioms.",
            "Anchor uyumluluğu **ayrı bir aksiyomdur** — Lie bracket aksiyomlarından türetilemez."
        ))
        if st.button(L("Prove","Kanıtla"), key="cc_anchor"):
            reg = PropertyRegistry()
            E = Symbol("E")
            A = LieAlgebroid(E, bracket=LieBracket(name="[·,·]_E"),
                             anchor=Anchor(name="ρ"), name="E-algebroid")
            X, Y = VectorFields("X Y", registry=reg)
            obs = A.anchor_compatibility_obstruction(X, Y, reg)
            st.write(L("Obstruction:","Obstrüksiyon:"), f"`{to_ascii(obs)}`")
            chain = A.prove_anchor_compatibility(X, Y, registry=reg)
            show_chain(chain, "anchor_compat")
            st.info(f"provenance: `{chain.steps[0].provenance_tag}`")

# ══════════════════════════════════════════════════════════════
# TAB 2 — Generalized Geometry  (spec §14)
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.header(L("Generalized Geometry", "Genelleştirilmiş Geometri"))
    st.markdown(L(
        "Courant algebroid, Dorfman & Courant brackets, H-twist, Dirac structures (spec §14).",
        "Courant algebroid, Dorfman & Courant bracket'ları, H-twist, Dirac yapıları (spec §14)."
    ))
    from jacopy.brackets.dorfman import SectionPair
    from jacopy.library.courant_algebroid import CourantAlgebroid
    from jacopy.library.dirac import DiracStructure, poisson_dirac, presymplectic_dirac

    sub2 = st.radio(L("Section","Bölüm"), [
        L("§14b–c Dorfman & Courant brackets","§14b–c Dorfman & Courant bracket'ları"),
        L("§14d Bridge identity","§14d Köprü kimliği"),
        L("§14a Courant Jacobi (untwisted)","§14a Courant Jacobi (bükümsüz)"),
        L("§14e H-twisted Courant","§14e H-bükülmüş Courant"),
        L("§14a Dirac structures","§14a Dirac yapıları"),
    ], key="gg_sub")

    reg_gg = PropertyRegistry()
    X_gg, Y_gg, Z_gg = VectorFields("X Y Z", registry=reg_gg)
    alpha_gg, beta_gg, gamma_gg = Forms("α β γ", degree=1, registry=reg_gg)
    a_gg = SectionPair(X_gg, alpha_gg)
    b_gg = SectionPair(Y_gg, beta_gg)
    C_gg = CourantAlgebroid()

    if L("§14b","§14b") in sub2:
        st.latex(r"[a,b]_D: \text{ Dorfman (Leibniz)}")
        st.latex(r"[a,b]_C: \text{ Courant (antisymmetric)}")
        if st.button(L("Expand","Aç"), key="gg_brackets"):
            dorf = C_gg.expand_dorfman(a_gg, b_gg, registry=reg_gg)
            cour = C_gg.expand(a_gg, b_gg, registry=reg_gg)
            st.subheader(L("Dorfman [a,b]_D","Dorfman [a,b]_D"))
            c1, c2 = st.columns(2)
            c1.write(L("vector:","vektör:")); c1.code(to_ascii(dorf.vector))
            c2.write(L("form:","form:")); c2.code(to_ascii(dorf.form))
            c2.latex(to_latex(dorf.form))
            st.subheader(L("Courant [a,b]_C","Courant [a,b]_C"))
            c1, c2 = st.columns(2)
            c1.write(L("vector:","vektör:")); c1.code(to_ascii(cour.vector))
            c2.write(L("form:","form:")); c2.code(to_ascii(cour.form))
            c2.latex(to_latex(cour.form))

    elif L("Bridge","Köprü") in sub2:
        st.latex(r"[a,b]_D - [a,b]_C = \left(0,\;\tfrac{1}{2}d(\iota_X\beta+\iota_Y\alpha)\right)")
        if st.button(L("Prove bridge","Köprü kanıtla"), key="gg_bridge"):
            chain = C_gg.prove_courant_dorfman_bridge(a_gg, b_gg, registry=reg_gg)
            show_chain(chain, "bridge")
            corr = C_gg.bridge_correction(a_gg, b_gg)
            st.write(L("Correction form:","Düzeltme formu:"))
            st.code(to_ascii(corr.form))
            st.latex(to_latex(corr.form))

    elif L("Jacobi (untwisted)","Jacobi (bükümsüz)") in sub2:
        st.latex(r"\operatorname{Jac}_C(a,b,c)=0")
        if st.button(L("Prove","Kanıtla"), key="gg_jac"):
            chain = C_gg.prove_jacobi_reduction(registry=reg_gg)
            show_chain(chain, "courant_jac")

    elif L("H-twisted","H-bükülmüş") in sub2:
        st.latex(r"\operatorname{Jac}_{C^H}(a,b,c)=0 \quad (dH=0)")
        if st.button(L("Prove","Kanıtla"), key="gg_htwist"):
            reg_h = PropertyRegistry()
            H = Symbol("H"); reg_h.declare(H, Graded(degree=3))
            X_h, Y_h, Z_h = VectorFields("X Y Z", registry=reg_h)
            alpha_h, beta_h = Forms("α β", degree=1, registry=reg_h)
            C_H = CourantAlgebroid(background_H=H)
            dorf_h = C_H.expand_dorfman(SectionPair(X_h,alpha_h), SectionPair(Y_h,beta_h), registry=reg_h)
            st.write(L("H-twisted Dorfman form half:","H-bükülmüş Dorfman form yarısı:"))
            st.code(to_ascii(dorf_h.form))
            chain = C_H.prove_jacobi_reduction(registry=reg_h)
            show_chain(chain, "htwist_jac")

    else:  # Dirac
        st.latex(r"L \subset TM\oplus T^*M,\quad \text{maximally isotropic, involutive}")
        if st.button(L("Show","Göster"), key="gg_dirac"):
            L_sym = Symbol("L")
            D = DiracStructure(C_gg, L_sym)
            pairing = D.pairing(a_gg, b_gg)
            iso_obs = D.isotropy_obstruction(a_gg)
            st.write(f"**Dirac:** {D.name}")
            c1, c2 = st.columns(2)
            c1.write(L("Pairing ⟨a,b⟩:","Eşleşme ⟨a,b⟩:"))
            c1.code(to_ascii(pairing)); c1.latex(to_latex(pairing))
            c2.write(L("Isotropy ⟨a,a⟩:","İzotropi ⟨a,a⟩:"))
            c2.code(to_ascii(iso_obs)); c2.latex(to_latex(iso_obs))
            chain_iso = D.prove_isotropy(a_gg)
            chain_inv = D.prove_involutivity(a_gg, b_gg)
            st.write(f"Isotropy: {len(chain_iso)} step, rule=`{chain_iso.steps[0].rule}`")
            st.write(f"Involutivity: {len(chain_inv)} step, rule=`{chain_inv.steps[0].rule}`")
            pi_d = Bivector("π", registry=reg_gg)
            omega_d = Symbol("ω"); reg_gg.declare(omega_d, Graded(degree=2))
            L_pi = poisson_dirac(pi_d, courant=C_gg)
            L_om = presymplectic_dirac(omega_d, courant=C_gg)
            st.write(f"Poisson Dirac L_π: `{L_pi.name}`")
            st.write(f"Presymplectic Dirac L_ω: `{L_om.name}`")

# ══════════════════════════════════════════════════════════════
# TAB 3 — Metric-Affine Geometry  (spec §11)
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.header(L("Metric-Affine Geometry", "Metrik-Affin Geometri"))
    st.markdown(L(
        "Connection, torsion, curvature, Bianchi identities, Cartan structure equations (spec §11).",
        "Bağlantı, torsiyon, eğrilik, Bianchi kimlikleri, Cartan yapı denklemleri (spec §11)."
    ))
    from jacopy.calculus.connection import AffineConnection, koszul_connection
    from jacopy.calculus.torsion_curvature import Torsion, Curvature
    from jacopy.calculus.local_frame import LocalFrame
    from jacopy.calculus.cartan_forms import ConnectionForm, TorsionForm, CurvatureForm
    from jacopy.library.bianchi_problem import BianchiProblem
    from jacopy.library.cartan_structure import CartanStructureProblem

    sub3 = st.radio(L("Section","Bölüm"), [
        L("§11b–g Connection, Torsion & Curvature","§11b–g Bağlantı, Torsiyon & Eğrilik"),
        L("§11i–k Connection/Torsion/Curvature forms","§11i–k Bağlantı/Torsiyon/Eğrilik formları"),
        L("§11s Bianchi identities","§11s Bianchi kimlikleri"),
        L("§11t Cartan structure equations","§11t Cartan yapı denklemleri"),
        L("§11p Difference of connections","§11p Bağlantı farkı"),
    ], key="ma_sub")

    nabla = AffineConnection("∇")
    X_ma = Derivation("X", 0)
    Y_ma = Derivation("Y", 0)
    Z_ma = Derivation("Z", 0)
    W_ma = Derivation("W", 0)
    reg_ma = PropertyRegistry()

    if L("§11b","§11b") in sub3:
        st.latex(r"T(X,Y):=\nabla_X Y - \nabla_Y X - [X,Y]")
        st.latex(r"R(X,Y)W:=\nabla_X\nabla_Y W - \nabla_Y\nabla_X W - \nabla_{[X,Y]}W")
        nabla_XY = nabla.eval(X_ma, Y_ma)
        T = Torsion(nabla, X_ma, Y_ma)
        R = Curvature(nabla, X_ma, Y_ma, W_ma)
        st.write(f"**§11b  ∇_X Y =** `{to_ascii(nabla_XY)}`")
        st.latex(to_latex(nabla_XY))
        c1, c2 = st.columns(2)
        c1.write("**§11f  T(X,Y):**"); c1.code(to_ascii(T)); c1.latex(to_latex(T))
        c2.write("**§11g  R(X,Y)W:**"); c2.code(to_ascii(R)); c2.latex(to_latex(R))
        st.write(L("**Four connection axioms:**","**Dört bağlantı aksiyomu:**"))
        axioms = [
            r"\nabla_{X+Y}Z = \nabla_X Z + \nabla_Y Z",
            r"\nabla_{fX}Y = f\nabla_X Y",
            r"\nabla_X(Y+Z) = \nabla_X Y + \nabla_X Z",
            r"\nabla_X(fY) = X(f)\cdot Y + f\nabla_X Y",
        ]
        for ax in axioms: st.latex(ax)

    elif L("forms","formları") in sub3:
        F = LocalFrame("F", dim=3)
        omega_ab = ConnectionForm(nabla, F, "a", "b")
        T_a      = TorsionForm(nabla, F, "a")
        R_ab     = CurvatureForm(nabla, F, "a", "b")
        st.write("**§11i  Connection 1-form ω^a_b(∇):**")
        st.code(str(omega_ab)); st.latex(to_latex(omega_ab))
        st.write("**§11j  Torsion 2-form T^a(∇):**")
        st.code(str(T_a)); st.latex(to_latex(T_a))
        st.write("**§11k  Curvature 2-form R^a_b(∇):**")
        st.code(str(R_ab)); st.latex(to_latex(R_ab))

    elif L("Bianchi","Bianchi") in sub3:
        st.latex(r"\operatorname{cycl}_{X,Y,Z} R(X,Y)Z = \operatorname{cycl}_{X,Y,Z}[(\nabla_X T)(Y,Z)+T(T(X,Y),Z)]")
        st.latex(r"\operatorname{cycl}_{X,Y,Z}(\nabla_X R)(Y,Z)W = \operatorname{cycl}_{X,Y,Z} R(X,T(Y,Z))W")
        if st.button(L("Prove both","İkisini de kanıtla"), key="ma_bianchi"):
            prob = BianchiProblem(nabla, registry=reg_ma)
            st.write(f"Engine rules: {len(prob.engine.definitions)}")
            res1 = prob.prove_first_bianchi(X_ma, Y_ma, W_ma)
            res2 = prob.prove_second_bianchi(X_ma, Y_ma, W_ma, Z_ma)
            st.success(L("Both proved ✅","İkisi de kanıtlandı ✅"))
            st.write(f"**Bianchi I:** ok={res1.ok}, lhs steps={len(res1.lhs_steps)}, rhs steps={len(res1.rhs_steps)}")
            st.write(f"**Bianchi II:** ok={res2.ok}, lhs steps={len(res2.lhs_steps)}, rhs steps={len(res2.rhs_steps)}")

    elif L("Cartan structure","Cartan yapı") in sub3:
        st.latex(r"T^a = de^a + \sum_b \omega^a_b \wedge e^b \quad \text{(I)}")
        st.latex(r"R^a_b = d\omega^a_b + \sum_c \omega^a_c \wedge \omega^c_b \quad \text{(II)}")
        eq = st.radio(L("Equation","Denklem"), ["Cartan I", "Cartan II"], key="ma_cartan_eq")
        if st.button(L("Prove","Kanıtla"), key="ma_cartan"):
            F = LocalFrame("F", dim=3)
            prob = CartanStructureProblem(nabla, F)
            U = Derivation("U", 0); V = Derivation("V", 0)
            st.write(f"Engine rules: {len(prob.engine.definitions)}")
            if eq == "Cartan I":
                lhs = prob.first_cartan_lhs(U, V, "a")
                rhs = prob.first_cartan_rhs(U, V, "a")
                st.write(f"LHS: `{to_ascii(lhs)}`"); st.write(f"RHS: `{to_ascii(rhs)}`")
                res = prob.prove_first_cartan(U, V, "a")
                st.success(L(f"ok={res.ok}, steps={len(res.steps)}",
                             f"ok={res.ok}, adım={len(res.steps)}"))
            else:
                lhs = prob.second_cartan_lhs(U, V, "a", "b")
                rhs = prob.second_cartan_rhs(U, V, "a", "b")
                st.write(f"LHS: `{to_ascii(lhs)}`"); st.write(f"RHS: `{to_ascii(rhs)}`")
                res = prob.prove_second_cartan(U, V, "a", "b")
                st.success(L(f"ok={res.ok}, steps={len(res.steps)}",
                             f"ok={res.ok}, adım={len(res.steps)}"))

    else:  # difference
        st.latex(r"\Delta(\nabla,\nabla')(X,Y):=\nabla_X Y - \nabla'_X Y")
        st.markdown(L(
            "Δ is C∞(M)-linear in both entries (follows from linearity axioms of both connections).",
            "Δ her iki girdide de C∞(M)-doğrusaldır (her iki bağlantının doğrusallık aksiyomlarından)."
        ))
        nabla2 = AffineConnection("∇'")
        delta = Sum(nabla.eval(X_ma, Y_ma), Neg(nabla2.eval(X_ma, Y_ma)))
        st.write(f"**Δ(∇,∇')(X,Y) =** `{to_ascii(delta)}`")
        st.latex(to_latex(delta))

# ══════════════════════════════════════════════════════════════
# TAB 4 — Symplectic & Poisson  (spec §12)
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.header(L("Symplectic & Poisson Geometry", "Simplektik & Poisson Geometri"))
    st.markdown(L(
        "Poisson bivector, Koszul bracket, tilde calculus, Jacobi reduction (spec §12).",
        "Poisson bivektör, Koszul bracket, tilde kalkülüs, Jacobi redüksiyonu (spec §12)."
    ))
    from jacopy.library.poisson import PoissonBracket
    from jacopy.library.symplectic import SymplecticManifold
    from jacopy.brackets.koszul import KoszulBracket
    from jacopy.calculus.musical import Sharp
    from jacopy.calculus.tilde import (
        tilde_interior, tilde_d, tilde_lie,
        tilde_intrinsic_engine, prove_tilde_cartan_relation,
        TildeDSquaredPoissonDefinition,
    )
    from jacopy.proof.expansion import ExpansionEngine as EE

    sub4 = st.radio(L("Section","Bölüm"), [
        L("§12a Symplectic manifold","§12a Simplektik manifold"),
        L("§12c Poisson bracket: 3 views","§12c Poisson bracket: 3 görünüm"),
        L("§12c Jacobi → [π,π]_SN=0","§12c Jacobi → [π,π]_SN=0"),
        L("§12g–h Koszul bracket","§12g–h Koszul bracket"),
        L("§12j–l Tilde calculus","§12j–l Tilde kalkülüs"),
    ], key="sp_sub")

    reg4 = PropertyRegistry()
    pi4 = Bivector("π", registry=reg4); reg4.declare(pi4, Poisson())
    (omega4,) = Forms("ω", degree=2, registry=reg4)
    f4, g4, h4 = Functions("f g h", degree=-1, registry=reg4)
    alpha4, beta4, gamma4 = Forms("α β γ", degree=1, registry=reg4)
    poisson4 = PoissonBracket.from_bivector(pi4)

    if L("§12a","§12a") in sub4:
        st.latex(r"\omega^\flat \circ \pi^\sharp = \mathrm{id}")
        if st.button(L("Show","Göster"), key="sp_symp"):
            M = SymplecticManifold(omega4, bivector=pi4, name="(M,ω,π)")
            st.write(f"**flat (ω♭):** `{M.flat}`")
            st.write(f"**sharp (π♯):** `{M.sharp}`")
            st.write(f"**compatibility:** `{M.compatibility}`")
            st.markdown(L("**Hamiltonian equivalence proof:**","**Hamiltonian eşdeğerlik kanıtı:**"))
            chain = M.prove_hamiltonian_equivalence(f4, registry=reg4)
            show_chain(chain, "ham_equiv")

    elif L("3 views","3 görünüm") in sub4:
        if st.button(L("Expand all 3","Tüm 3'ü göster"), key="sp_3views"):
            d1 = poisson4.expand(f4, g4, reg4)
            d2 = poisson4.via_hamiltonian(f4, g4)
            d3 = poisson4.koszul_expand(alpha4, beta4, reg4)
            st.subheader(L("View 1 — Derived bracket","Görünüm 1 — Türetilmiş bracket"))
            show_expr("{f,g}_π", d1)
            st.subheader(L("View 2 — Hamiltonian vector field","Görünüm 2 — Hamiltonian vektör alanı"))
            show_expr("X_f(g)", d2)
            st.subheader(L("View 3 — Koszul (on forms)","Görünüm 3 — Koszul (formlarda)"))
            show_expr("{α,β}_π", d3)

    elif L("[π,π]","[π,π]") in sub4:
        st.latex(r"\operatorname{Jac}_{\{\cdot,\cdot\}_\pi}(f,g,h)\Longrightarrow[\pi,\pi]_{SN}=0")
        if st.button(L("Prove reduction","Redüksiyonu kanıtla"), key="sp_jac"):
            obs = poisson4.jacobi_obstruction(reg4)
            st.write(L("Obstruction:","Obstrüksiyon:"), f"`{to_ascii(obs)}`")
            st.latex(to_latex(obs))
            chain = poisson4.prove_jacobi_reduction(f4, g4, h4, registry=reg4)
            show_chain(chain, "poisson_jac")
            chain_k = poisson4.prove_koszul_jacobi_reduction(alpha4, beta4, gamma4, registry=reg4)
            st.markdown("---")
            st.write(L("**Form (Koszul) Jacobi:**","**Form (Koszul) Jacobi:**"))
            show_chain(chain_k, "koszul_jac")

    elif L("Koszul bracket","Koszul bracket") in sub4:
        st.latex(r"[\alpha,\beta]_K = L_{\pi^\sharp(\alpha)}\beta - L_{\pi^\sharp(\beta)}\alpha - d\langle\pi^\sharp(\alpha),\beta\rangle")
        if st.button(L("Expand","Aç"), key="sp_koszul"):
            sharp4 = Sharp(pi4)
            koszul4 = KoszulBracket(sharp4)
            koz_ab = koszul4.expand(alpha4, beta4)
            show_expr("[α,β]_K", koz_ab)
            st.write(L("**Algebroid properties:**","**Algebroid özellikleri:**"))
            st.write(f"is_graded_antisymmetric: `{koszul4.is_graded_antisymmetric}`")
            st.write(f"satisfies_leibniz: `{koszul4.satisfies_leibniz}`")
            st.write(f"satisfies_graded_jacobi: `{koszul4.satisfies_graded_jacobi}`")

    else:  # tilde calculus
        st.latex(r"\tilde{d}V = [\pi,V]_{SN},\quad \tilde{L}_\omega V = \tilde{d}\tilde{\iota}_\omega V + \tilde{\iota}_\omega\tilde{d}V")
        if st.button(L("Prove tilde magic","Tilde magic kanıtla"), key="sp_tilde"):
            reg_t = PropertyRegistry()
            pi_t = Bivector("π", registry=reg_t); reg_t.declare(pi_t, Poisson())
            eta_t = Symbol("η"); reg_t.declare(eta_t, Graded(degree=1))
            mu_t  = Symbol("μ"); reg_t.declare(mu_t,  Graded(degree=1))
            V_t   = Symbol("V"); reg_t.declare(V_t,   Graded(degree=1))
            W_t   = Symbol("W"); reg_t.declare(W_t,   Graded(degree=2))
            i_t = tilde_interior(eta_t)
            d_t = tilde_d(pi_t)
            L_t = tilde_lie(eta_t, pi_t)
            i_m = tilde_interior(mu_t)
            sharp_t = Sharp(pi_t)
            koz_t   = KoszulBracket(sharp_t)
            eng_t   = tilde_intrinsic_engine(pi_t, koz_t, sharp=sharp_t, registry=reg_t)
            lhs_m = Act(L_t, V_t)
            rhs_m = Sum(Act(d_t, Act(i_t, V_t)), Act(i_t, Act(d_t, V_t)))
            chain_m = prove_tilde_cartan_relation(lhs_m, rhs_m, etas=(eta_t,), engine=eng_t, registry=reg_t)
            st.success(L(f"Tilde magic proved in {len(chain_m)} steps ✅",
                         f"Tilde magic {len(chain_m)} adımda kanıtlandı ✅"))
            lhs_a = Sum(Act(i_t, Act(i_m, W_t)), Act(i_m, Act(i_t, W_t)))
            chain_a = prove_tilde_cartan_relation(lhs_a, Integer(0), etas=(eta_t,), engine=eng_t, registry=reg_t)
            st.success(L(f"Anti-commute proved in {len(chain_a)} steps ✅",
                         f"Anti-komütasyon {len(chain_a)} adımda kanıtlandı ✅"))
            engine_dsq = EE([TildeDSquaredPoissonDefinition(pi_t, registry=reg_t)])
            out_dsq, steps_dsq = engine_dsq.expand(Act(d_t, Act(d_t, V_t)))
            st.success(L(f"d̃²=0 proved: {steps_dsq[0].rule}",
                         f"d̃²=0 kanıtlandı: {steps_dsq[0].rule}"))

# ══════════════════════════════════════════════════════════════
# TAB 5 — Calculus on Algebroids  (spec §13)
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.header(L("Calculus on Algebroids", "Algebroid Kalkülüs"))
    st.markdown(L(
        "K and K̃ operators, derivator identities §3.1.5, compatibility conditions (spec §13).",
        "K ve K̃ operatörleri, §3.1.5 derivator kimlikleri, uyumluluk koşulları (spec §13)."
    ))
    from jacopy.calculus.cartan_remainder import K
    from jacopy.calculus.tilde import K_tilde, tilde_d, tilde_lie, tilde_interior
    from jacopy.calculus.derivator import derivator
    from jacopy.library.koszul_problem import KoszulProblem

    sub5 = st.radio(L("Section","Bölüm"), [
        L("§13b K and K̃ operators","§13b K ve K̃ operatörleri"),
        L("§13c–g Derivator identity (1) form","§13c–g Derivator kimliği (1) form tarafı"),
        L("§13c–g Derivator identity (1') multivector","§13c–g Derivator kimliği (1') çokvektör tarafı"),
        L("§13d Compatibility — same obstruction","§13d Uyumluluk — aynı obstrüksiyon"),
    ], key="alg_sub")

    if L("§13b","§13b") in sub5:
        st.latex(r"K_V := -\mathcal{L}_V + d\circ\iota_V")
        st.latex(r"\tilde{K}_\eta := -\tilde{\mathcal{L}}_\eta + \tilde{d}\circ\tilde{\iota}_\eta")
        reg5 = PropertyRegistry()
        pi5   = Symbol("π"); reg5.declare(pi5, Graded(degree=1)); reg5.declare(pi5, Poisson())
        omega5 = Symbol("ω"); reg5.declare(omega5, Graded(degree=1))
        eta5   = Symbol("η"); reg5.declare(eta5,   Graded(degree=1))
        U5     = Symbol("U"); reg5.declare(U5,     Graded(degree=1))
        K_U5 = K(U5)
        K_til5 = K_tilde(eta5, pi5)
        expr_K5    = Act(K_U5, omega5)
        expr_Ktil5 = Act(K_til5, U5)
        c1, c2 = st.columns(2)
        c1.write("**K_U(ω):**"); c1.code(to_ascii(expr_K5)); c1.latex(to_latex(expr_K5))
        c2.write("**K̃_η(U):**"); c2.code(to_ascii(expr_Ktil5)); c2.latex(to_latex(expr_Ktil5))
        st.write(f"K_U degree: `{K_U5._degree}`,  K̃_η degree: `{K_til5._degree}`")

    elif L("form","form tarafı") in sub5:
        st.latex(r"\mathcal{D}^{T^*M}_{\mathcal{L}_U}(\eta,\mu) = \mathcal{L}_{\tilde{K}_\eta U}\mu + K_{\tilde{K}_\mu U}\eta")
        if st.button(L("Prove (109 steps)","Kanıtla (109 adım)"), key="alg_der1"):
            from jacopy.calculus.lie_derivative import lie_derivative
            reg_d = PropertyRegistry()
            pi_d = Symbol("π"); eta_d = Symbol("η"); mu_d = Symbol("μ")
            U_d = Symbol("U"); V_d = Symbol("V"); W_d = Symbol("W")
            for s in (pi_d, eta_d, mu_d, U_d, V_d, W_d): reg_d.declare(s, Graded(degree=1))
            Y_d = Derivation("Y", 0)
            xi_d = Symbol("ξ"); reg_d.declare(xi_d, Graded(degree=1))
            prob_d = KoszulProblem(pi_d, (eta_d, mu_d, Symbol("ν")), registry=reg_d,
                                   multivectors=((U_d,1),(V_d,1),(W_d,1)))
            prob_d.assume_poisson()
            K_b_d = prob_d.koszul_bracket
            lhs_d = derivator(lie_derivative(U_d), K_b_d, eta_d, mu_d)
            Kte_U = Act(K_tilde(eta_d, pi_d), U_d)
            Ktm_U = Act(K_tilde(mu_d,  pi_d), U_d)
            rhs_d = Sum(Act(lie_derivative(Kte_U), mu_d), Act(K(Ktm_U), eta_d))
            chain_d = prob_d.prove_derivator(lhs_d, rhs_d, eval_args=(Y_d,), side="form")
            show_chain(chain_d, "derivator_1_form")

    elif L("multivector","çokvektör tarafı") in sub5:
        st.latex(r"\tilde{\mathcal{D}}^{SN}_{\tilde{\mathcal{L}}_\eta}(U,V) = \tilde{\mathcal{L}}_{\tilde{K}_U\eta}V + \tilde{K}_{\tilde{K}_V\eta}U")
        if st.button(L("Prove (117 steps)","Kanıtla (117 adım)"), key="alg_der1p"):
            from jacopy.brackets.schouten import sn as sn_mv
            reg_mv = PropertyRegistry()
            pi_mv = Symbol("π"); eta_mv = Symbol("η"); mu_mv = Symbol("μ")
            U_mv = Symbol("U"); V_mv = Symbol("V"); W_mv = Symbol("W")
            for s in (pi_mv, eta_mv, mu_mv, U_mv, V_mv, W_mv): reg_mv.declare(s, Graded(degree=1))
            Y_mv = Derivation("Y", 0)
            xi_mv = Symbol("ξ"); reg_mv.declare(xi_mv, Graded(degree=1))
            prob_mv = KoszulProblem(pi_mv, (eta_mv, mu_mv, Symbol("ν")), registry=reg_mv,
                                    multivectors=((U_mv,1),(V_mv,1),(W_mv,1)))
            prob_mv.assume_poisson()
            lhs_mv = derivator(tilde_lie(eta_mv, pi_mv), sn_mv, U_mv, V_mv)
            K_U_eta = Act(K(U_mv), eta_mv)
            K_V_eta = Act(K(V_mv), eta_mv)
            rhs_mv = Sum(Act(tilde_lie(K_U_eta, pi_mv), V_mv),
                         Act(K_tilde(K_V_eta, pi_mv), U_mv))
            chain_mv = prob_mv.prove_derivator(lhs_mv, rhs_mv, eval_args=(xi_mv,), side="multivector")
            show_chain(chain_mv, "derivator_1p_mv")

    else:  # compatibility
        st.markdown(L(
            "Function Jacobi and form (Koszul) Jacobi both reduce to the **same obstruction** `[π,π]_SN=0`.",
            "Fonksiyon Jacobi ve form (Koszul) Jacobi her ikisi de aynı **obstrüksiyona** indirgenir: `[π,π]_SN=0`."
        ))
        if st.button(L("Show","Göster"), key="alg_compat"):
            from jacopy.library.poisson import PoissonBracket as PB5
            reg_c = PropertyRegistry()
            pi_c = Bivector("π", registry=reg_c); reg_c.declare(pi_c, Poisson())
            f_c, g_c, h_c = Functions("f g h", degree=-1, registry=reg_c)
            al_c, be_c, ga_c = Forms("α β γ", degree=1, registry=reg_c)
            pb_c = PB5.from_bivector(pi_c)
            chain_f = pb_c.prove_jacobi_reduction(f_c, g_c, h_c, registry=reg_c)
            chain_k = pb_c.prove_koszul_jacobi_reduction(al_c, be_c, ga_c, registry=reg_c)
            c1, c2 = st.columns(2)
            c1.subheader(L("Function Jacobi","Fonksiyon Jacobi"))
            c1.write(f"rule: `{chain_f.steps[0].rule}`")
            c1.code(to_ascii(chain_f.steps[0].after))
            c2.subheader(L("Form (Koszul) Jacobi","Form (Koszul) Jacobi"))
            c2.write(f"rule: `{chain_k.steps[0].rule}`")
            c2.code(to_ascii(chain_k.steps[0].after))
            same = str(chain_f.steps[0].after) == str(chain_k.steps[0].after)
            if same:
                st.success(L("Same obstruction ✅","Aynı obstrüksiyon ✅"))
            else:
                st.error(L("Different!","Farklı!"))