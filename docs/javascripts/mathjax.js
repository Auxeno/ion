window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|jupyter-wrapper",
  },
};

document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    MathJax.typesetPromise();
  }
});
