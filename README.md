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
