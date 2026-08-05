# OptiBench experiment dashboard

The dashboard is a static paper-oriented view of completed OptiBench studies.
Its active sections cover CIFAR-10 depth scaling, CIFAR-100 dataset scaling,
momentum ablations, and character-level language modeling. Each experiment owns
an independently scrollable figure matrix while the zoom controls, metric
geometry, and full-resolution dialog are shared.

Rebuild every active figure from retrieved CSV bundles with:

```bash
../27-ICLR-OptiBench/.venv/bin/python scripts/build_dashboard_plots.py
```

The builder computes each row's limits from every curve in its section after
that section's configured moving average, then supplies those same limits to every
separate six-row figure. It writes the exact source paths and numeric limits to
`assets/plots/manifest.json`. Validation curves remain unsmoothed, and scale
error is always quadratic. Complete-window trailing simple moving average is
the default: CIFAR sections use SMA-100 and the character-level Shakespeare
section uses SMA-50 and displays task losses as perplexity. Exponential moving
average remains selectable with a directly specified decay.

The figures used by the original dashboard are retained verbatim under
`assets/archive/legacy-plots/` and are not referenced by the active page.
