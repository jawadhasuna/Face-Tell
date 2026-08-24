# Technical report

`Face-Tell-Technical-Report.pdf` — 7 pages, A4.

Regenerate after editing `paper.html`:

```
chrome --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="paper/Face-Tell-Technical-Report.pdf" \
  "file:///<abs-path>/paper/paper.html"
```

Edge works identically in place of Chrome. Print styling lives in the `@page`
and print rules at the top of `paper.html`; there is no separate stylesheet.
