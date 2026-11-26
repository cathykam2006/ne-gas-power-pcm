# Browser version of the app (GitHub Pages)

This branch holds one page, `index.html`, that runs the Streamlit app entirely
in the visitor's browser with [stlite](https://github.com/whitphx/stlite). There
is no server, so it never goes to sleep. The Streamlit Community Cloud app keeps
running from `main`.

- The page loads `app.py`, `src/negem/`, `scenarios/`, `data/` and `results/`
  from `main` each time it opens, so the daily live run committed by
  `live-daily.yml` appears here automatically.
- "Run the model live" works too: scipy's HiGHS solver runs in the browser.
  It is slower than on a server, so short windows are best.
- Every file in `FROM_MAIN` must exist on `main`. If the app starts reading a new
  file, add it there.
