<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Based on the info from this website: [https://user.eumetsat.int/resources/user-guides/mtg-irs-level-1-data-guide](https://user.eumetsat.int/resources/user-guides/mtg-irs-level-1-data-guide) I want to build a model of corrections for sentinel insar. I want to use the IRS information about water vapour and temperature. Make a model with mathematical definition

A practical way to use MTG‑IRS temperature and water‑vapour for Sentinel‑1 InSAR is to build a tropospheric phase‑delay correction model based on refractivity and radiative‑transfer kernels. Below is a compact but explicit mathematical formulation.

***

## 1. IRS temperature and humidity state

Let $x(\mathbf{r})$ be the atmospheric state at horizontal location $\mathbf{r}$ and height $z$.
From MTG‑IRS retrievals you obtain vertical profiles of temperature $T$ and specific humidity $q_v$ (or water‑vapour mixing ratio) on pressure levels. The Level‑1 IRS spectra are converted to Level‑2 geophysical profiles by an inversion using Jacobians of radiance with respect to temperature and humidity.[^1][^2][^3]

Define the state vector at location $\mathbf{r}$:

$$
x(\mathbf{r}) =
\begin{bmatrix}
T(z_1,\mathbf{r}) \\ \vdots \\ T(z_N,\mathbf{r}) \\
q_v(z_1,\mathbf{r}) \\ \vdots \\ q_v(z_N,\mathbf{r})
\end{bmatrix}
$$

with associated error covariance matrix $S_x$ provided by the IRS retrieval scheme (or approximated from profile uncertainties).[^3][^1]

***

## 2. Tropospheric refractivity from $T,q_v$

For C‑band Sentinel‑1, the neutral‑atmosphere refractivity $N$ at height $z$ can be expressed as a function of pressure $P$, temperature $T$, and water‑vapour partial pressure $e$.[^4][^5]

Use a standard formula (e.g. Saastamoinen‑type, adapted to radar):

1. Convert specific humidity $q_v(z)$ to vapour pressure $e(z)$:

$$
e(z) = \frac{q_v(z)}{0.622 + 0.378\,q_v(z)}\,P(z)
$$

2. Refractivity $N(z)$ (in N‑units):

$$
N(z) = k_1 \frac{P(z)}{T(z)} + k_2 \frac{e(z)}{T(z)} + k_3 \frac{e(z)}{T(z)^2}
$$

where $k_1,k_2,k_3$ are empirically determined microwave refractivity constants for C‑band.[^5][^4]

Split into hydrostatic and wet parts:

$$
N_h(z) = k_1 \frac{P_d(z)}{T(z)}, \qquad
N_w(z) = k_2 \frac{e(z)}{T(z)} + k_3 \frac{e(z)}{T(z)^2}
$$

with $P_d = P - e$.

***

## 3. Mapping to slant delay and phase

For a Sentinel‑1 acquisition at incidence angle $\theta$, the slant path $s$ can be parameterised with height $z$ assuming straight rays. The wet slant delay for acquisition $i$ at point $\mathbf{r}$ is

$$
L_w^{(i)}(\mathbf{r}) = 10^{-6} \int_{0}^{z_{\text{top}}}
N_w\big(z,\mathbf{r}, t_i\big)\, m(z,\theta)\, \mathrm{d}z
$$

where $m(z,\theta)$ is a mapping factor from vertical to slant (for small bending, $m \approx 1/\cos\theta$).[^4][^5]
The corresponding radar phase delay at wavelength $\lambda$ is

$$
\phi_w^{(i)}(\mathbf{r}) =
\frac{4\pi}{\lambda} L_w^{(i)}(\mathbf{r})
$$

Similarly, you can include the hydrostatic component $L_h^{(i)}$ if desired; usually the hydrostatic part is more spatially smooth and often removed by long‑wavelength filtering in InSAR.[^5][^4]

***

## 4. Differential InSAR atmospheric phase screen

For an interferogram formed by acquisitions at times $t_1$ and $t_2$, the atmospheric phase screen (APS) at pixel $\mathbf{r}$ is

$$
\Delta \phi_{\text{atm}}(\mathbf{r}) =
\phi_w^{(2)}(\mathbf{r}) - \phi_w^{(1)}(\mathbf{r})
+ \phi_h^{(2)}(\mathbf{r}) - \phi_h^{(1)}(\mathbf{r})
$$

Using the IRS‑derived refractivity fields, this becomes

$$
\Delta \phi_{\text{atm}}(\mathbf{r}) =
\frac{4\pi}{\lambda} 10^{-6}
\int_{0}^{z_{\text{top}}}
\big[
\Delta N_w(z,\mathbf{r}) + \Delta N_h(z,\mathbf{r})
\big]\, m(z,\theta)\, \mathrm{d}z
$$

with

$$
\Delta N_{w,h}(z,\mathbf{r}) =
N_{w,h}(z,\mathbf{r}, t_2) - N_{w,h}(z,\mathbf{r}, t_1)
$$

computed from the two IRS 3‑D fields closest in time to the Sentinel‑1 acquisitions.[^6][^4][^5]

The corrected interferometric phase is then

$$
\phi_{\text{corr}}(\mathbf{r}) =
\phi_{\text{obs}}(\mathbf{r}) - \Delta \phi_{\text{atm}}(\mathbf{r})
$$

where $\phi_{\text{obs}}$ is the measured wrapped phase after removing topography.[^4][^5]

***

## 5. IRS‑constrained variational refinement (optional)

Because IRS spatial/temporal sampling may not match Sentinel‑1 exactly, you can cast the APS estimation as a regularised inversion constrained by IRS:

1. Represent the unknown vertical wet‑refractivity field at time $t_i$ as $N_w(z,\mathbf{r}, t_i)$ on a 3‑D grid.
2. Define a cost function combining IRS and InSAR information:

$$
J(N_w) = 
\big\| H_{\text{IRS}}(N_w) - y_{\text{IRS}} \big\|_{S_{\text{IRS}}^{-1}}^2
+
\big\| G(N_w) - \Delta \phi_{\text{obs}} \big\|_{S_{\phi}^{-1}}^2
+
\big\| N_w - N_{w,\text{b}} \big\|_{S_b^{-1}}^2
$$

- $H_{\text{IRS}}$ maps refractivity (or profiles of $T,q_v$) to IRS radiances or Level‑2 retrieval space using the same forward model and Jacobians as in the IRS Level‑1/2 ATBD.[^3]
- $y_{\text{IRS}}$ are the IRS radiances or retrieved profiles.
- $G$ maps $N_w$ to the differential phase through the integral in Section 4 for both epochs.
- $N_{w,\text{b}}$ is a background field (e.g. from NWP).
- $S_{\text{IRS}}, S_{\phi}, S_b$ are covariance matrices weighting the three terms.[^2][^1][^3]

3. Minimise $J$ (e.g. Gauss–Newton) to obtain an IRS‑ and InSAR‑consistent wet‑refractivity field $\hat{N}_w$, then compute $\Delta \phi_{\text{atm}}$ as above.

This variational approach exploits the high vertical sensitivity of IRS together with the high horizontal resolution of Sentinel‑1 to derive a more realistic APS than from either data source alone.[^1][^2][^6][^5]

***

If you tell me your preferred vertical coordinate (height vs pressure) and whether you have IRS Level‑2 profiles directly, I can rewrite the model explicitly in that coordinate and with discrete summations ready for implementation.
<span style="display:none">[^10][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://dl.iafastro.directory/event/IAC-2022/paper/69040/

[^2]: https://indico.esa.int/event/93/sessions/823/attachments/2792/3243/P4_DallAmico.pdf

[^3]: https://user.eumetsat.int/s3/eup-strapi-media/MTG_IRS_Level_1_Algorithm_Theoretical_Basis_Document_v2_7b42d5efaa.pdf

[^4]: https://ui.adsabs.harvard.edu/abs/2018ISPAr42.3..741K/abstract

[^5]: https://earth.esa.int/eogateway/documents/20142/37627/Atmospheric_water_vapour_determination_InSAR_GNSS.pdf

[^6]: https://www.sciencedirect.com/science/article/pii/S2667393222000096

[^7]: https://irs.ctbto.org/static-assets/IRS User Guide ver 1.2.pdf

[^8]: https://www.irs.gov/pub/irs-pdf/p5717.pdf

[^9]: https://www-cdn.eumetsat.int/files/2022-03/Product User Guide – IASI level2 TS, T, Q release 1_1.pdf

[^10]: https://database.eohandbook.com/database/instrumentsummary.aspx?instrumentID=950

