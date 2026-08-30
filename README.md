## Lightcurve Fourier Decomposition
Direct Fourier decomposition for multi-band Cepheid variable light curves
- Find initial period by Lomb-Scargle method (implemented by <a href=https://www.astroml.org/gatspy/ style="color: black">`gatspy`</a>([VanderPlas et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract))
- Iterative penalized least-squares fitting
- Multi-band compilation of Fourier parameters is also available 

This code is adopted to analyze the [OGLE IV](https://ogledb.astrouw.edu.pl/~ogle/OCVS/) light curves of Cepheid/RR Lyrae variables.

## Fit reliability and uncertainty

`FourierDecomp.uncertainty` contains two uncertainty levels:

- `conditional_invariant_uncertainty`: fast HC3 covariance with period and
  Fourier order fixed.  This is a conditional lower bound.
- `bootstrap_fourier_decomp`: epoch/transit bootstrap that re-runs period
  search, order selection, optimization, clipping, and amplitude refitting.

For Gaia, the bootstrap automatically resamples matched G/BP/RP rows by
`transit_id` when it is available.  Use at least 100 replicates for the formal
catalog summary and more replicates for high-priority candidates.

```python
from FourierDecomp import decomposition, uncertainty
from FourierDecomp.IO import wire_globals

# Keep the existing notebook setup: load ls_data/df_ident, then wire globals.
wire_globals(decomposition, ls_data, df_ident, df_rrfit=df_rrfit,
             templates=templates)

result = uncertainty.bootstrap_fourier_decomp(
    source_id,
    mode="gaia",
    n_boot=100,
    random_state=20260827,
    decomp_kwargs={
        "period_fit": False,
        "use_optim": True,
        "adaptive_lam": True,
        "use_refit": True,
    },
    return_replicates=False,
)
```

On a CPU server, parallelize over source IDs and keep the replicate loop inside
each source serial.  Also limit BLAS threads to one per worker to avoid nested
parallelism.

## Catalog fit, deep refit, and merge

The default blind search remains a single-term Lomb--Scargle search with
`K=5` candidate peaks and no automatic harmonic expansion.  The implementation
now applies the following safeguards:

- the upper search period is capped below the source time baseline;
- only observed, scientifically activated bands enter the periodogram;
- the initial Fourier order is capped by the available residual degrees of
  freedom instead of forcing `M_MAX=15` on sparse light curves; and
- `K` and `harmonics` supplied to `fourier_decomp` or `mp_run` are honored.

Deep refitting is a separate pass.  `audit_period_stability_catalog` can screen
the nominal catalog with a deeper `K=15` single-term search and low-order robust
fits to explicit integer period families.  The screen is a triage diagnostic,
not a period estimator: an unresolved alias remains `review`, and absence of a
warning is not proof that the period is unique.  Sources without an active band
or enough degrees of freedom are structural failures and are not repeatedly
refitted.

Use `merge_refit_error_catalogs` to create the final second-output catalog.  It
never edits the base catalog, rejects non-finite HC3 rows and unresolved refit
QA, enforces one row per ID, and writes a separate decision audit.  A recovered
missing source is appended; an existing source is replaced only by an accepted
refit.  The complete executable sequence is in
`Fourier_Decomposition_Reliability_Workflow.ipynb`.

The reference-free period audit was checked against the available OGLE LMC
catalog periods as an exploratory regression: with the conservative defaults,
it selected about 68% of the discrepant solutions in that comparison while
flagging 4% of a 50-source agreement control sample.  Therefore it must be
combined with bootstrap/candidate-family probabilities and an abstention tier;
it must not be interpreted as a complete alias detector.
