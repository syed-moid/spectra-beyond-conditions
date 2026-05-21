# Phase 0 — Literature Grounding for Microscopic Variability in INS of Defect Ferroelectrics

*v0.3 — incorporates three rounds of physics review. Sharpens §1.8 ML-bridge,
§1.9 ensemble-broadening mechanism, fixes concrete ℓ_q parametrization for
Phase 1, and tightens ξ₂ asymmetric window with skew-sampling.*

## 0. Executive summary

The literature on inelastic neutron scattering (INS) and adjacent
spectroscopies (Brillouin, hyper-Raman, IR, RUS) of BaTiO₃ and isostructural
perovskites provides strong empirical support for the existence of
microscopic, sample-dependent variability in the phonon response that is
**not captured** by macroscopic (T, c, E). The variability is physically
substantial — the soft E(TO) mode in tetragonal BaTiO₃ at room temperature
is deeply overdamped (Γ/ω₀ ≈ 3), and damping differs categorically between
single crystals and ceramics of nominally identical composition. The
dominant microphysical sources are (i) eight-site off-center Ti disorder
and chain correlations, (ii) TA-TO mode coupling with q- and zone-dependent
linewidth Γ(q,T), (iii) polar nanoregion / precursor cluster populations,
(iv) oxygen vacancy concentration and clustering, (v) domain-wall and
twin-boundary scattering, (vi) random local strain fields, (vii) effective
anharmonic coefficient scatter, (viii) central-peak ↔ soft-mode spectral
weight redistribution, and (ix) distributed relaxation timescales /
non-Lorentzian lineshape distortion.

Each of these produces spectral observables — linewidths, asymmetry,
central-peak structure, mode-coupling lineshapes — that are observed to
vary across nominally equivalent samples at the few-tens-of-percent level,
with order-unity tails in specific regimes (ceramic vs single-crystal
damping; CP presence/absence in nominally equivalent BaTiO₃ thin films).
**The latent-microphysics reformulation has a defensible empirical
foundation.** Recommended Phase 1 instantiation: a primary multiplicative
linewidth perturbation ξ₁ with spatial correlation across q (initial
correlation length ℓ_q = 3 q-bins, swept over {1, 3, 5, 8}), augmented by
a secondary scalar strain perturbation ξ₂ drawn from a skew-normal
distribution with asymmetric support [−5%, +7%]. Both enter through the
existing Green's-function self-energy machinery; M is recomputed from the
perturbed realized spectrum (no direct M perturbation). Initial linewidth
perturbation scale: δΓ/Γ₀ ∈ 10-40%, with sensitivity sweeps to larger
tails. PNR / spectral-weight redistribution and distributed-relaxation
mechanisms are deferred to Phase 2.

---

## 1. Pass 1: Mechanisms — what drives microscopic variability in ferroelectric INS spectra

This section identifies the physical mechanisms documented across the
perovskite ferroelectric family (BaTiO₃, SrTiO₃, KNbO₃, PbTiO₃, KTaO₃, and
relaxors PMN, PZN, KTN, BST). Material-specific magnitudes are deferred to
Pass 2.

### 1.1 Eight-site off-center Ti / chain correlation disorder

The off-center model of Comes, Lambert & Guinier (1968) — supplemented by
NMR (Zalar, Blinc 2005), EXAFS (Ravel & Stern 1998), and pair distribution
function studies (Senn et al. 2016) — establishes that even in nominally
cubic paraelectric BaTiO₃ and KNbO₃, the Ti ion is *locally* off-centered
along ⟨111⟩ with chain-like correlations along ⟨100⟩ that persist hundreds
of degrees above Tc. The crystallographic cubic symmetry emerges from
disordered averaging of these local rhombohedral-like distortions. NMR
observation of nonzero quadrupole coupling at Ti sites in cubic BaTiO₃
*and* SrTiO₃ directly demonstrates off-center sites even where macroscopic
susceptibility shows pure displacive behavior.

**Spectral consequences:** (i) sheet-like diffuse scattering on {100}
planes; (ii) a relaxational central peak in Raman and Brillouin spectra
below Td but above Tc; (iii) overdamped soft TO modes in BaTiO₃ in sharp
contrast to well-defined soft modes in PbTiO₃ (Harada-Axe-Shirane 1971;
Tomeno 2020). The chain correlation length is largely an intrinsic feature
of BaTiO₃ rather than a strongly sample-tunable variable; for that reason
it is deferred as a primary latent-variable axis (see §4).

### 1.2 TA-TO mode coupling

In BaTiO₃, KNbO₃, PMN, PZN, and KTN, the lowest TA and soft TO branches
anticross in specific Brillouin zone regions. The coupling is parameterized
by a wave-vector- and temperature-dependent TO linewidth Γ(q, T), which
controls both the position and lineshape of the apparent TA peak (Hlinka
et al., Stock et al. 2002, Gehring et al.). The "phonon waterfall" feature
in relaxors is the visible manifestation: below a critical q_wf the TO
mode disappears into an overdamped feature that depends on the Brillouin
zone (i.e. on structure factors, not just on dynamics).

**Spectral consequences:** broadening and asymmetry of TA peaks in zones
where the structure factor mixes them with TO; appearance/disappearance of
waterfall features; lineshape interference between modes. Note that
variability in TA-TO coupling strength is implicitly modulated by
variability in Γ(q, T) itself — once ξ₁ (linewidth) and ξ₂ (frequency
shift) are in play, explicit modeling of coupling-strength variability
becomes redundant for Phase 1.

### 1.3 Polar nanoregion (PNR) / precursor cluster volume fraction

Below the Burns temperature Td (a sample property, not strictly intrinsic),
small regions of nonreversible local polarization appear within the
otherwise nonpolar matrix. Most extensively documented in relaxors (PMN:
Td ≈ 620 K; PZN; KTN), but well-established in canonical BaTiO₃:
Brillouin LA softening (Ko et al.), RUS Vogel-Fulcher fits with activation
energy ~0.2 eV (Salje et al.), SHG below ~580 K with power-law exponent
crossover, and ultrafast X-ray speckle dynamics all show precursor polar
clusters between Td ≈ 580 K and Tc ≈ 403 K. PNR concentration depends on
defects, strain, and growth conditions.

**Spectral consequences:** (i) central peak intensity above Tc; (ii)
anomalous TA damping (Stock et al. PMN, Gvasaliya KTN); (iii) coupling of
polarization fluctuations to LA via density fluctuations; (iv) soft-mode
"recovery" in PMN below 220 K when PNR dynamics freeze.

**Scope note:** Modeling PNR explicitly requires a two-component
(matrix + cluster) spectral decomposition, which substantially complicates
the generator. Deferred to Phase 2.

### 1.4 Oxygen vacancy concentration, clustering, and defect dipoles

V_O is the dominant native defect in BaTiO₃, SrTiO₃, and most perovskite
ferroelectrics. First-principles and TEM studies (Park et al. 2017;
Cordero anelastic spectroscopy; Tyunina et al. 2020) show V_O are rarely
isolated: divacancies and linear V_O chains along [001] are energetically
preferred, especially under tensile strain. Around each cluster, lattice
distortion extends over several unit cells. V_O can act as a donor and
form defect dipoles (V_O-B cation) that pin domain walls. Multiple charge
states (+1, +2, neutral) modulate dipole strengths.

**Spectral consequences:** optical-spectroscopy studies of SrTiO₃₋δ
document broadening of phonon modes with increasing V_O density and a
V_O-induced mid-IR band whose strength scales with carrier concentration.
In BaTiO₃₋δ the soft TO mode shows a pronounced drop in energy upon
electron doping and splits into low/high components. For nominally
stoichiometric BaTiO₃, V_O concentrations are ~10⁻⁴ to 10⁻³, and residual
variations at this level produce few-tens-of-percent linewidth shifts.
This is a primary contributor to ξ₁.

### 1.5 Domain walls and twin-boundary scattering

180° and 90° ferroelectric domain walls and ferroelastic twin boundaries
are universal in ferroelectrics, and their density and mobility depend
strongly on growth, mechanical history, and electric/thermal cycling.
Bugallo et al. (2021) deconvolved thermal-conductivity data on a
composition-spread PbTiO₃ film and showed ferroelectric DWs are *at least
as effective as point defects* in scattering phonons, even at point-defect
levels ~15%. RUS on BaTiO₃ ceramics vs single crystals shows damping is
"largest in the tetragonal phase of ceramic materials but very low in
single crystals" — a categorical difference attributable to mobile twin
boundaries.

**Spectral consequences:** linewidth enhancement, low-frequency tails,
modified Q factors in resonant ultrasonic spectroscopy, and
frequency-dependent dielectric loss. This is the dominant contributor to
order-unity damping variability when comparing ceramic vs single-crystal
samples and is part of the empirical basis for an order-unity tail in the
ξ₁ distribution.

### 1.6 Random local strain fields from substitutional disorder

In solid solutions — Ba(Ti,Zr)O₃, Ba(Ti,Sn)O₃, (Ba,Sr)TiO₃,
(Na,Bi)TiO₃-BaTiO₃ — B-site cation size mismatch generates random elastic
fields whose statistics depend on local composition fluctuations. Laulhé
et al. attributed the relaxor state of BaTi₁₋ₓZrₓO₃ to random elastic
fields from Ti-O₆ / Zr-O₆ size mismatch. Even isovalent doping introduces
a continuous distribution of local TO frequencies and consequently a
distribution of linewidths via inhomogeneous broadening.

**Pure-BTO scope note:** the strongest published magnitudes for
strain-induced soft-mode shifts come from chemically disordered systems
(BST, BZT) and strained epitaxial films. For nominally pure bulk BaTiO₃,
local strain *exists* (driven by residual defects, sub-grain mosaic,
thermal stress) but its distribution is less cleanly characterized in the
literature. This is reflected in §4 by ξ₂ being a secondary perturbation
with a deliberately modest amplitude window.

### 1.7 Higher-order anharmonic coupling fluctuations

Even in defect-free crystals, third- and fourth-order anharmonic coupling
produces a self-energy whose temperature dependence is well-described by
Cowley anharmonic lattice dynamics. However, when overlapped with the
disorder mechanisms above, the *effective* anharmonic coefficients
estimated from Raman/INS lineshapes show sample-to-sample scatter — Ouni
et al. 2016 documented this for tetragonal BaTiO₃, and an arXiv survey
"Revisiting 70 years of lattice dynamics of BaTiO₃" notes "much larger
width is observed for phonons in BaTiO₃" compared to PbTiO₃ and
K₀.₅Bi₀.₅TiO₃ despite similar formal anharmonicity.

### 1.8 Central-peak ↔ soft-mode spectral weight transfer

A subtle but physically important effect that is *not equivalent* to pure
linewidth broadening: near Tc and below Td, some samples display strong
central-peak emergence accompanied by reduced spectral weight in the
propagating overdamped mode, while in other nominally equivalent samples
the propagating channel retains most of its weight. This is a
redistribution of intensity between the relaxational and propagating
channels of the dynamical susceptibility χ"(q, ω), not a broadening of
either channel individually. The CP-present/CP-absent dichotomy across
BaTiO₃ and (Ba,Sr)TiO₃ thin films (§2.3) is the cleanest empirical
signature.

**Spectral consequences:** integrated intensity ratio I(CP)/I(DHO),
spectral asymmetry at intermediate ω, and the shape of the low-ω wing.
Crucially, this information is **not captured by (T, c, E)** — two samples
at identical macroscopic conditions can have different partitioning — and
is **richer than what ξ₁ alone can encode**. In representation-learning
terms, this mechanism alters the relative distribution of spectral weight
across low-ω and propagating components without requiring large shifts in
nominal peak position or linewidth, making it a potentially
information-rich axis for spectrum-based models — one that metadata-only
models structurally cannot represent because the partitioning is
realization-specific rather than condition-specific. Deferred to Phase 2
alongside explicit PNR modeling, but flagged as a potentially high-impact
latent axis for the second paper in the sequence.

### 1.9 Distributed relaxation times / non-Lorentzian lineshape distortion

The DHO and Green's-function framework as currently implemented assumes a
single effective damping scale Γ(q, T) per mode. In real samples near
precursor regimes, however, relaxation is rarely single-timescale:
distributed local environments (V_O clusters of varying size, PNRs of
varying volume, locally strained microregions) produce a *distribution* of
relaxation times. The spectral signature is non-Lorentzian — typical
phenomenological fits use stretched-exponential / Kohlrausch forms or
sum-of-Lorentzians. Hlinka's two-overlapping-mode picture for cubic BaTiO₃
is a discrete-approximation manifestation of this continuous distribution.

**Spectral consequences:** lineshape asymmetry, slow tails at high
|ω − ω₀|, deviations from DHO that grow as one approaches Tc. Critically,
this is a mechanism where **spectrum-based models can outperform DHO
baselines in principle**, because the DHO functional form itself is wrong
while the spectrum carries the correct (non-Lorentzian) shape information.
This is highly relevant for the eventual ST-vs-DHO comparison in the
paper.

**Phase 1 treatment.** A full Kohlrausch / stretched-exponential
implementation is not required for Paper 1. Instead, the spatial-
correlation structure of the primary ξ₁ perturbation already generates
non-Lorentzianity through an explicit physical mechanism: convolving
locally varying damping realizations across correlated microenvironments
produces an effective ensemble susceptibility that deviates from a
single-Lorentzian DHO form, generating broadened wings and asymmetric
tails analogous to distributed-relaxation phenomenology. This is enough
for Phase 1 falsification. An explicit distributed-relaxation generator
is flagged for Phase 2.

### 1.10 Experimental / instrumental convolution variability

Real INS data carries sample- and instrument-specific contributions that
are not microscopic physics but are real latent variability from the
modeler's perspective: mosaic spread, grain texture in polycrystalline
samples, finite energy/momentum resolution of the spectrometer, focusing
geometry, and detector-specific kernel shape. The existing augmentation
framework already partially captures these (resolution broadening,
additive noise). For taxonomic completeness they are listed here as a
distinct category — the latent-microphysics reformulation is concerned
primarily with §1.1-1.9, and §1.10 remains the responsibility of the
augmentation pipeline rather than the data generator.

---

## 2. Pass 2: BaTiO₃-specific magnitudes

This section narrows from mechanisms to quantitative magnitudes drawn from
BaTiO₃ measurements where available, with SrTiO₃ and KNbO₃ as supporting
evidence.

### 2.1 Soft-mode linewidth in BaTiO₃ — order-of-magnitude variability

The most striking single fact: the soft E(TO) mode in tetragonal BaTiO₃
at room temperature has ω₀ ≈ 35 ± 5 cm⁻¹ and FWHM Γ ≈ 100 ± 10 cm⁻¹
(Hlinka et al., reviewed in *Ferroelectrics* 375, 2008). This is
**deeply overdamped** — Γ/ω₀ ≈ 3. The damping varies systematically with
T (increases toward Tc) and additionally varies across sample classes and
growth methods. Reported INS and Raman linewidths for nominally pure
single-crystal BaTiO₃ at fixed (T, E = 0) scatter at the
few-tens-of-percent level across published studies, with substantially
larger spread (order-unity) when comparing ceramic to single-crystal
samples or across thin-film growth methods. In single-domain tetragonal
BaTiO₃ (Harada-Axe 1970), the soft mode along [110] with e ∥ [1̄10]
extrapolates to 4.5 meV at the zone center, with phonons well-defined
except at very small q (≤ 0.05 Å⁻¹) where they become overdamped. In
*cubic* BaTiO₃ (Tomeno 2020, T = 453 K, ~50 K above Tc), the TO branches
along [100] and [111] soften toward Γ and the [110] branch with [001]
polarization is "relatively low and flat … overdamped." Inelastic diffuse
scattering intensity along [100] and [110] grows on cooling toward Tc —
i.e. dynamic Ti disorder coexists with soft modes.

The two-mode picture (Hlinka): the full soft-mode response in the cubic
phase consists of **two overlapping overdamped modes** in the 1-150 cm⁻¹
range — a critical CP-like mode at 10-50 cm⁻¹ and a partial-softening
mode at 60-100 cm⁻¹ (cubic), 200-280 cm⁻¹ (tetragonal). The relative
weight is sample-dependent and reflects the order-disorder vs displacive
balance.

### 2.2 SrTiO₃ — quantum paraelectric baseline

Yamada-Shirane 1969 measured the soft optical phonon energy width grow
from **0.22 meV at 20 K to 0.85 meV at 250 K** — roughly a factor of 4
increase over the full temperature range. The dispersion is strongly
anisotropic around q = 0. The soft mode hardens substantially under
applied E-field (Worlock-Fleury, Akimov 2000) and under epitaxial strain
— thin films show "markedly different behavior of the soft modes … from
that in the bulk … explained by the existence of local polar regions"
(Sirenko, *Nature* 2000). Frequency modulation depth of ~20% with bias
voltage is reproducibly demonstrated.

### 2.3 Central peak — sample-dependent presence/absence

In BaTiO₃/Ba₀.₃Sr₀.₇TiO₃ superlattices, "the appearance of the central
peak is not a general rule. For instance, the overdamped soft mode was
studied by Raman scattering in BaTiO₃ and Ba₁₋ₓSrₓTiO₃ thin films but no
central peak was observed." Two nominally similar BaTiO₃ samples can
therefore display qualitatively different spectra — one with a clean
overdamped DHO, the other with a coexisting CP + DHO. This is one of the
clearest pieces of evidence for latent variability beyond (T, c, E) and
is the canonical empirical motivation for mechanism §1.8 (spectral weight
transfer) being a Phase 2 priority.

### 2.4 Burns temperature spread

T_d in BaTiO₃ is reported in the range 560-620 K depending on the probe
(SHG: 580 K; RUS: 586 K; piezoelectric: variable). T_c is well-pinned at
403-405 K. The ~175 K precursor regime is dominated by polar cluster
dynamics whose density correlates with crystal quality.

### 2.5 Domain-wall contribution magnitude

For PbTiO₃ thin films across a composition-spread (Pb-deficient to
Pb-rich, ±20%), the ferroelectric DW contribution to phonon scattering
rivals or exceeds the point-defect contribution at concentrations up to
15%. For BaTiO₃, the RUS damping in the tetragonal phase of ceramic
samples qualitatively exceeds that in single crystals; this category
provides the empirical basis for the order-unity *tail* of the ξ₁
distribution. The bulk of the distribution — applicable to "research-grade
single-crystal" samples — lies at the few-tens-of-percent level.

### 2.6 Defect-induced broadening in BaTiO₃₋δ and SrTiO₃₋δ

Highly reduced SrTiO₃₋δ (down to SrTiO₂.₇₂) shows phonon broadening that
increases with V_O density and a V_O-induced mid-IR absorption whose
strength scales with carrier concentration. In BaTiO₃₋δ the soft TO mode
in the rhombohedral phase splits into low- and high-frequency components,
with the high-frequency component recovering to undoped BaTiO₃ values at
low δ. For typical "stoichiometric" samples, V_O concentrations are
~10⁻⁴ to 10⁻³, and residual variations at this level are consistent with
few-tens-of-percent linewidth shifts — the central magnitude target for
Phase 1.

---

## 3. Variability taxonomy table

Magnitude column framed as physically plausible spread ranges with their
direct literature anchors, not universal multiplicative factors. Phase
column indicates whether each mechanism is implemented in Phase 1,
deferred to Phase 2, subsumed into another mechanism, or handled by the
augmentation pipeline.

| # | Latent mechanism (ξ) | Physical origin | Spectral observable | Plausible magnitude in BaTiO₃ family | Phase | Key references |
|---|---|---|---|---|---|---|
| 1 | Eight-site Ti chain correlations | Off-center Ti ions in TiO₆ cage, chain-correlated along ⟨100⟩ with disorder between chains | {100} diffuse sheets; relaxational CP at 10-50 cm⁻¹ in cubic BaTiO₃; overdamping of TO along [100] | Largely intrinsic to BaTiO₃; modest sample-to-sample variation | Defer | Comes-Lambert-Guinier (1968); Pirc-Blinc (2004); Zalar NMR (2005); Ravel EXAFS (1998); Senn (2016); Tomeno (2020) |
| 2 | TA-TO mode coupling | Anharmonic + structure-factor mediated mixing of TA and soft TO | Dispersion anticrossing; TA peak asymmetry; waterfall features; Γ(q,T) | Modulated implicitly by ξ₁ and ξ₂; not separately tunable in Phase 1 | Implicit | Harada-Axe-Shirane (1971); Stock (2002); Hlinka; Gvasaliya KTN (2010) |
| 3 | PNR / precursor cluster volume fraction | Local correlated Ti displacements producing nm-scale polar regions below T_d | CP intensity above T_c; anomalous TA damping; acoustic softening; low-T TO "recovery" in relaxors | T_d ≈ 580 K, T_c ≈ 403 K in BaTiO₃; PNR fraction at room T varies with preparation; RUS Vogel-Fulcher activation ~0.2 eV | Phase 2 | Burns-Dacol (1983); Wada SHG (2012); Gehring (2001); Salje RUS; Stock-Gehring PMN (2002); Ko Brillouin |
| 4 | Oxygen vacancy density + clustering | V_O point defects; divacancies and [001] chains; V_O-B defect dipoles | Phonon broadening scaling with [V_O]; mid-IR band; soft mode splitting in BaTiO₃₋δ | Few-tens-of-percent linewidth broadening for δ ~10⁻³; full soft-mode splitting at δ ~10⁻² in BaTiO₃₋δ | Phase 1 (component of ξ₁) | Cordero series; Tyunina (2020); SrTiO₃₋δ optical; Park JPCL (2017) |
| 5 | Domain wall / twin boundary density | 180°/90° ferroelectric walls and ferroelastic twins; mobility under stress/field | Low-frequency damping enhancement; RUS Q reduction; thermal-conductivity reduction comparable to ~15% point-defect concentration | Order-unity damping difference between ceramic and single-crystal BaTiO₃ in tetragonal phase | Phase 1 (tail of ξ₁) | Salje RUS BaTiO₃; Bugallo PbTiO₃ ACS AMI (2021); Catalan-Schlom |
| 6 | Local strain / random elastic field | B-site size mismatch; defect-induced strain; epitaxial strain in films | Soft mode frequency shift; inhomogeneous broadening; film-bulk differences | Quantitatively established in solid solutions and films; in nominally pure BTO, modest residual contribution | Phase 1 (ξ₂, secondary) | Laulhé BaTi(Zr)O₃; Sirenko *Nature* (2000); Tyunina; Senn (2016) |
| 7 | Anharmonic coefficient effective scatter | Third/fourth-order phonon self-energy with sample-dependent prefactors emerging from disorder | Higher-frequency optical modes show larger widths in BaTiO₃ than chemically similar perovskites | A₁/E TO mode widths in tetragonal BaTiO₃ vary at the few-tens-of-percent level across published studies | Subsumed in ξ₁ | Ouni (2016); Cowley anharmonic theory; "Revisiting 70 years" review (2020) |
| 8 | CP ↔ soft-mode spectral weight transfer | Redistribution of intensity between relaxational and propagating channels of χ"(q,ω) | I(CP)/I(DHO) ratio; spectral asymmetry; low-ω wing shape | CP presence/absence across nominally equivalent BaTiO₃ thin films; not captured by (T,c,E) | Phase 2 | Hlinka *Ferroelectrics* 375 (2008); thin-film Raman literature (§2.3) |
| 9 | Distributed relaxation times / non-Lorentzian broadening | Distribution of local environments → distribution of damping scales | Non-Lorentzian lineshapes; slow tails; deviations from DHO that grow toward Tc | Hlinka two-overlapping-mode picture is the discrete-approximation manifestation | Phase 1 (via ξ₁ spatial correlation); Phase 2 explicit | Hlinka (2008); Cowley generalized self-energy |
| 10 | Experimental convolution variability | Mosaic, texture, finite spectrometer resolution, detector kernel | Resolution-broadened lineshapes, instrument-specific noise | Sample-dependent; already partially captured in augmentation pipeline | Augmentation domain, not generator | — (handled in existing augmentation framework) |

---

## 4. Phase 1 candidate ranking and recommendation

Ranking against the design principles in the handoff — (a) latent variable
must affect spectrum non-trivially, (b) must not be deterministically
captured by (T, c, E), (c) must have literature-derived magnitudes ≥
10-20%, and (d) must be physically grounded:

**Rank 1 — Local TO linewidth perturbation ξ₁ driven by effective
disorder strength.** Strongest literature support across mechanisms §1.4
(V_O), §1.5 (DW), §1.7 (anharmonic scatter), with §1.9 (distributed
relaxation) emerging naturally from spatial structure of ξ₁. Maps cleanly
into the existing Green's-function self-energy framework as a
multiplicative modification of Σ"(ω, q). *Mechanistic interpretation:*
defect scattering rate adds to the intrinsic anharmonic rate; the realized
spectrum incorporates the perturbed Γ, and M is recomputed from the
perturbed realized spectrum — no direct M perturbation.

**Rank 2 — Local strain field ξ₂ (frequency-shift perturbation).**
Physically essential for solid solutions and epitaxial films. For
nominally pure bulk BaTiO₃ at fixed nominal c, this variable couples
mostly into mode frequencies via small TO frequency shifts. Best
implemented as a small secondary perturbation augmenting Rank 1.

**Rank 3 — PNR volume fraction.** Strong literature support but
implementation complexity is substantial (matrix + cluster two-component
spectra, CP-DHO coupling). Deferred to Phase 2.

**Rank 4 — Spectral weight transfer (CP ↔ DHO).** Closely tied to Rank 3
mechanistically. Deferred to Phase 2.

**Rank 5 — TA-TO coupling strength variability.** Implicitly modulated
once ξ₁ and ξ₂ are in play; explicit modeling adds little for Paper 1.

**Rank 6 — Eight-site chain correlation length.** Universal feature of
BaTiO₃, not strongly sample-tunable as a continuous variable. Deferred.

### Recommended Phase 1 minimal viable reformulation

Implement **one** physically-grounded latent variable family combining the
dominant Rank 1 effect with a small Rank 2 contribution.

#### Primary perturbation: ξ₁ — multiplicative, spatially correlated linewidth perturbation

$$\Gamma_\text{realized}(q, T;\, \xi_1) = \Gamma_0(q, T)\,\bigl[1 + \alpha\,\xi_1(q)\bigr]$$

where:

- **Γ₀(q, T)** is the existing literature-calibrated DHO width from the
  Shirane-anchored framework.
- **ξ₁(q)** is a smooth per-sample random field over q with spatial
  correlation length ℓ_q. Implementation: a low-pass-filtered Gaussian
  random field, or equivalently a Gaussian process with a smoothing kernel
  of width ℓ_q. The smoothness requirement is physical: real disorder
  broadens phonons coherently over correlated q-regions, not point-by-point.
  Spatial correlation across ω at fixed q can be enforced via post-
  convolution with a narrow Gaussian over ω.
- **α** is the per-sample amplitude scale, drawn from a distribution
  calibrated so that the ensemble RMS of α·ξ₁ produces realized δΓ/Γ₀ ∈
  10-40% as the central regime, with order-unity tails accessible via
  heavier-tailed sampling (e.g. log-normal with appropriate σ) representing
  the ceramic/single-crystal extreme.

**Correlation length ℓ_q.** Treated as a tunable Phase 1 hyperparameter
with physical meaning. Initial value:

$$\ell_q = 3 \text{ q-bins (default)}$$

Sensitivity sweep:

$$\ell_q \in \{1, 3, 5, 8\} \text{ q-bins}$$

The lower bound ℓ_q = 1 approaches white-noise / iid perturbation
(localized disorder), the upper bound ℓ_q = 8 represents extended
microstructural correlation. The default ℓ_q = 3 is anchored to the
q-grid resolution of the existing generator and matches the natural
bandwidth of physically-resolved phonon broadening. **Warning:** ℓ_q must
not be made too large, or ξ₁ degenerates into an effectively global
scaling factor — which would reintroduce metadata-like degeneracy and
defeat the purpose of the reformulation.

**Multiplicative vs additive form (revised from v0.1).** Multiplicative
form is preferred because it (i) preserves positivity of Γ automatically,
(ii) is physically more realistic — defect-scattering and anharmonic rates
compose as rates rather than as additive offsets — and (iii) behaves
better numerically in the Γ/ω ≳ 1 overdamped regime characteristic of
BaTiO₃.

#### Secondary perturbation: ξ₂ — scalar local strain, frequency-shift form

$$\omega_{0,\,\text{realized}}(q;\, \xi_2) = \omega_0(q)\,\bigl[1 + \beta\,\xi_2\bigr]$$

with ξ₂ a per-sample scalar drawn from a **skew-normal distribution** on
the asymmetric support:

$$\beta\,\xi_2 \in [-5\%, +7\%]$$

The asymmetric window reflects physical asymmetries in the BaTiO₃ family:
tensile distortions soften the TO mode more strongly than compressive
distortions harden it; compressive hardening saturates earlier; precursor
softening is asymmetric near Tc; and defect-induced local volume expansion
is more common than strong local compression. The upper bound is reduced
from the v0.1 estimate of +10% to +7% to remain physically conservative
in the nominally pure BaTiO₃ regime, given that ξ₁ already carries the
bulk of the latent-information burden.

**Skew-normal sampling rather than hard-clipped Gaussian.** A symmetric
Gaussian would impose E[ξ₂] = 0, encouraging cancellation effects in
ensemble averages and producing unrealistically balanced synthetic
spectra. Real microstructural perturbation distributions are rarely
symmetric. A skew-normal (or log-skewed) sampling distribution introduces
mild but realistic asymmetry that reflects defect-skewed real materials.

#### M reconstruction

M is recomputed from the perturbed realized spectrum via the existing
pipeline. **The latent variables affect M only through their effect on
the spectrum, never directly.** This is the critical design principle
from the handoff and is what makes the reformulation an inverse problem
that spectrum-based models can solve in principle while metadata-only
models structurally cannot.

#### Acceptance criterion

(Re-stating the Phase 1 plan): spectrum-only Spectral Transformer
achieves MAE_logM at least 30% below the nonlinear-conditions MLP floor
(0.0048) under realistic variability of (ξ₁, ξ₂) at the 10-40% central
regime with ℓ_q = 3. If achieved, Option 3 (latent-microphysics
reformulation) is validated and Phase 2 proceeds. If not, run sensitivity
sweeps:

1. ℓ_q sweep over {1, 3, 5, 8} at fixed α distribution
2. α distribution sweep to larger amplitudes (order-unity tails)
3. ξ₂ skewness sweep

before concluding. The goal is to find the variability scale at which
spectral information becomes decisive, which is itself a publishable
finding.

---

## 5. Risks, caveats, and counterarguments

**Caveat 1 — The "variability is substantial" claim relies primarily on
optical spectroscopies, not INS itself.** Most quantitative spread numbers
come from Raman, Brillouin, IR ellipsometry, and RUS. Direct INS
measurements on BaTiO₃ are sparse and primarily on high-quality single
crystals (Harada, Yamada, Shirane, Tomeno). The implicit claim is that
the INS spectrum reflects the same underlying microphysics — physically
defensible but worth flagging in the manuscript.

**Caveat 2 — Some "variability" may reflect unresolved physics rather
than true sample variability.** The displacive vs order-disorder
controversy in BaTiO₃ is decades-old and unresolved. What looks like
sample variability could partly be measurement variability or model-
dependent extraction. The Phase 1 latent-variable approach is robust to
this — we don't need to resolve the controversy, only to acknowledge that
the spectrum carries information not captured by (T, c, E).

**Caveat 3 — Magnitudes were extracted opportunistically.** Several
values in the taxonomy table (especially the "few-tens-of-percent" and
"order-unity tail" language) are synthesis claims rather than single
peer-reviewed numbers. Before fixing α and β values in the Phase 1
generator, a focused re-read of Tomeno (2020), Hlinka (2008), and
Stock-Gehring is warranted to anchor the distribution parameters to
specific datasets.

**Caveat 4 — Counterargument: maybe variability *is* well-explained by
(T, c, E).** A strict reading is hard to sustain given the Sirenko thin-
film/bulk difference and the ceramic-vs-single-crystal damping data, but
a Bayesian skeptic could argue: "Those examples are *different* (T, c, E),
not the same conditions with hidden microphysics." Counter-counter: the
SrTiO₃ Yamada-Shirane scatter, the BaTiO₃ central-peak presence/absence
in nominally identical thin films (§2.3), and the well-documented sample-
quality dependence of soft-mode damping at fixed nominal (T, c, E = 0)
directly contradict this objection.

**Caveat 5 — DHO functional form is itself an approximation.** Once
distributed relaxation (§1.9) is taken seriously, the DHO baseline is
fitting a wrong functional form to non-Lorentzian data. This is actually
a *positive* asymmetry for the paper: spectrum-based models can exploit
lineshape information the DHO cannot represent. The Phase 1 generator
implicitly captures this through smooth ξ₁(ω) realizations producing mild
non-Lorentzian distortion via ensemble averaging over correlated
microenvironments.

**Verdict:** The literature supports the latent-microphysics reformulation
with sufficient confidence to proceed to Phase 1, with the implementation
design above.

---

## 6. References

- Comes, Lambert & Guinier, *Solid State Commun.* **6**, 715 (1968) —
  chain structure of BaTiO₃ and KNbO₃, eight-site model origin.
- Yamada & Shirane, *J. Phys. Soc. Jpn.* **26**, 396 (1969) — SrTiO₃
  neutron scattering, linewidth 0.22 → 0.85 meV (20-250 K).
- Hüller, *Z. Phys.* **220**, 145 (1969) — soft-mode dispersion model
  for BaTiO₃; dynamical interpretation of diffuse sheets.
- Harada, Axe & Shirane, *Phys. Rev. B* **4**, 155 (1971) —
  neutron-scattering reinvestigation of BaTiO₃ soft optic phonons;
  TA-TO coupling.
- Burns & Dacol, *Solid State Commun.* **48**, 853 (1983) — Burns
  temperature and polar nanoregion concept.
- Ravel, Stern & Vedrinskii, *Ferroelectrics* **206**, 407 (1998) — EXAFS
  confirmation of eight-site disorder in BaTiO₃.
- Sirenko et al., *Nature* **404**, 373 (2000) — soft-mode hardening in
  SrTiO₃ thin films, local polar regions.
- Stock et al., *Phys. Rev. B* **65**, 144101 (2002); Gehring et al.,
  neutron studies of PMN soft TO mode and PNRs.
- Pirc & Blinc, *Phys. Rev. B* **70**, 134107 (2004) — off-center Ti
  model of BaTiO₃.
- Zalar et al., *Phys. Rev. B* **71**, 064107 (2005) — NMR evidence for
  off-center Ti in cubic BaTiO₃ and SrTiO₃.
- Hlinka et al., *Ferroelectrics* **375**, (2008) — two-overlapping-modes
  picture of cubic BaTiO₃ soft response.
- Wada et al., *Phys. Rev. B* (2012) — SHG evidence for broken local
  symmetry above Tc in BaTiO₃.
- Senn et al., *Phys. Rev. Lett.* (2016) — local symmetry-breaking
  distortions and chain correlations in BaTiO₃ from PDF.
- Ouni, Chapron, Aroui et al., *Appl. Phys. A* **122**, 480 (2016) —
  anharmonic analysis of tetragonal BaTiO₃ optical phonons.
- Park et al., *J. Phys. Chem. Lett.* **8**, 3409 (2017) — oxygen vacancy
  linear clustering in SrTiO₃.
- Tomeno, Fernandez-Baca, Chi, Oka & Tsunoda, *J. Phys. Soc. Jpn.* **89**,
  054601 (2020) — INS at T = 453 K in cubic BaTiO₃.
- Bugallo et al., *ACS Appl. Mater. Interfaces* **13**, 45679 (2021) —
  DW vs point-defect phonon scattering in PbTiO₃ thin films.
- Tyunina et al., *Phys. Rev. Res.* (2020) — oxygen vacancy dipoles in
  strained epitaxial BaTiO₃.
- "Revisiting 70 years of lattice dynamics of BaTiO₃," arXiv:2012.12669
  (2020) — anharmonicity and Raman width comparisons.

---

## Status and next steps

**Phase 0: complete.** The literature foundation is sufficient to proceed
to Phase 1 implementation without the project feeling speculative. The
framework now has a physically meaningful reason why spectra can contain
information not recoverable from metadata alone, the perturbations are
grounded, the ML task is identifiable, and the falsification criterion is
honest.

### Recommended next-step sequence

1. **Commit v0.3 to `paper/phase0_literature.md` in the repo** — this
   document is the empirical backbone of the latent-microphysics
   reformulation and should be in version control before Phase 1
   implementation begins.

2. **Focused parameter-anchoring extraction.** Re-read Hlinka (2008) and
   Tomeno (2020) — both available open-access — to extract specific
   Γ values at specific (T, q) points. Use these to anchor the α
   distribution (linewidth amplitude) in the Phase 1 generator. Without
   this anchoring step, the "10-40%" central regime is a defensible
   estimate; with it, it becomes a directly literature-calibrated number
   that survives reviewer scrutiny.

3. **Phase 1 generator design changes.** Three concrete code changes
   needed in `src/data/generator/`:
   - Inject ξ₁(q) sampling with smoothing kernel of width ℓ_q (default 3
     q-bins) into the linewidth calculation.
   - Inject ξ₂ skew-normal scalar into the zone-center frequency.
   - Recompute M from the perturbed realized spectrum (no code change to
     the M calculation itself; the input spectrum just changes).

4. **Re-run the diagnostic from Session 7.** Re-train the
   nonlinear-conditions MLP floor on the new ξ-augmented data (it should
   stay roughly at 0.0048 since the conditions are unchanged), then train
   the spectrum-only ST and check whether MAE_logM falls below the floor
   by ≥30%. This is the Phase 1 acceptance gate.

5. **If Phase 1 succeeds: proceed to Phases 2-4** (full implementation,
   manuscript drafting, submission). If it fails: run the three
   sensitivity sweeps before concluding the reformulation is insufficient.
