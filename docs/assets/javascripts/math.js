/* The official Zensical document$ observable also fires after instant navigation. */
document$.subscribe(({ body }) => {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    throwOnError: false,
  });
});
