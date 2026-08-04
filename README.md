# OptiBench experiment dashboard

The dashboard is a static paper-oriented view of completed OptiBench studies.
Each experiment owns an independently scrollable figure matrix while the zoom
controls, metric geometry, and full-resolution dialog are shared.

Rebuild every active figure from retrieved CSV bundles with:

```bash
../27-ICLR-OptiBench/.venv/bin/python scripts/build_dashboard_plots.py
```

The builder computes each row's limits from every curve in its section after
that section's configured moving average, then supplies those same limits to every
separate six-row figure. It writes the exact source paths and numeric limits to
`assets/plots/manifest.json`. Validation curves remain unsmoothed, and scale
error is always quadratic. Exponential moving average is the default, with
`decay = epsilon ** (1 / window_size)` and `epsilon = 0.01`; simple trailing
moving average remains selectable. CIFAR sections use EMA-100; the
character-level Shakespeare section uses EMA-10 and displays task losses as
perplexity.

The figures used by the original dashboard are retained verbatim under
`assets/archive/legacy-plots/` and are not referenced by the active page.
