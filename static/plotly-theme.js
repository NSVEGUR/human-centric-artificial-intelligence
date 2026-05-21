/* =============================================================
   Plotly dark theme override.
   Wraps Plotly.newPlot and Plotly.react so every chart in the app
   gets dark paper/plot backgrounds, light fonts, and subtle grid
   lines that match the global design tokens. No Python touched.
   ============================================================= */
(function () {
  if (!window.Plotly) {
    console.warn("plotly-theme.js: Plotly not loaded yet");
    return;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
  }

  function hsl(token, alpha) {
    var v = cssVar(token);
    if (!v) return "rgba(0,0,0,0)";
    return alpha != null ? "hsl(" + v + " / " + alpha + ")" : "hsl(" + v + ")";
  }

  function darkAxis() {
    return {
      gridcolor: hsl("--border"),
      zerolinecolor: hsl("--border-strong"),
      linecolor: hsl("--border"),
      tickcolor: hsl("--border"),
      tickfont: { color: hsl("--muted-foreground") },
      title: { font: { color: hsl("--foreground") } },
    };
  }

  function baseLayout() {
    return {
      paper_bgcolor: hsl("--card"),
      plot_bgcolor: hsl("--card"),
      font: {
        color: hsl("--foreground"),
        family:
          'Geist, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif',
        size: 12,
      },
      colorway: [
        "#60a5fa",
        "#a78bfa",
        "#34d399",
        "#fbbf24",
        "#f87171",
        "#22d3ee",
        "#f472b6",
        "#a3e635",
      ],
      xaxis: darkAxis(),
      yaxis: darkAxis(),
      legend: {
        bgcolor: "rgba(0,0,0,0)",
        font: { color: hsl("--foreground") },
        bordercolor: hsl("--border"),
      },
      hoverlabel: {
        bgcolor: hsl("--popover"),
        bordercolor: hsl("--border"),
        font: { color: hsl("--foreground"), family: "Geist, system-ui" },
      },
      margin: { t: 40, r: 20, b: 40, l: 50 },
    };
  }

  /* Deep-merge so user layouts (e.g. xaxis.title.text) keep their
     own keys while inheriting the dark theme defaults. */
  function deepMerge(target, source) {
    if (!source) return target;
    Object.keys(source).forEach(function (key) {
      var src = source[key];
      if (
        src &&
        typeof src === "object" &&
        !Array.isArray(src) &&
        target[key] &&
        typeof target[key] === "object" &&
        !Array.isArray(target[key])
      ) {
        target[key] = deepMerge(Object.assign({}, target[key]), src);
      } else {
        target[key] = src;
      }
    });
    return target;
  }

  /* For multi-axis figures (xaxis2, yaxis2, ...) make sure they also
     get the dark axis defaults. */
  function applyAxesDefaults(layout) {
    Object.keys(layout || {}).forEach(function (k) {
      if ((k.startsWith("xaxis") || k.startsWith("yaxis")) && k !== "xaxis" && k !== "yaxis") {
        layout[k] = deepMerge(deepMerge({}, darkAxis()), layout[k] || {});
      }
    });
    return layout;
  }

  ["newPlot", "react"].forEach(function (fn) {
    var orig = window.Plotly[fn].bind(window.Plotly);
    window.Plotly[fn] = function (el, data, layout, config) {
      var merged = deepMerge(baseLayout(), layout || {});
      merged = applyAxesDefaults(merged);
      var finalConfig = Object.assign(
        { responsive: true, displaylogo: false },
        config || {}
      );
      return orig(el, data, merged, finalConfig);
    };
  });
})();
