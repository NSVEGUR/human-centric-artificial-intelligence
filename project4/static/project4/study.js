/* Project 4 — shared movie-card rendering + trial controllers.
 *
 * Posters are best-effort: fetched client-side from Wikipedia's public,
 * keyless, CORS-enabled REST API. If the lookup fails, times out, or the
 * browser has no network access at all, the card simply keeps its
 * generated gradient placeholder — the study remains fully usable offline.
 */

const P4 = {};

// ── gradient placeholder (deterministic hash of the title) ──────────────
P4.gradientForTitle = function (title) {
  let hash = 0;
  for (let i = 0; i < title.length; i++) {
    hash = (hash * 31 + title.charCodeAt(i)) >>> 0;
  }
  const h1 = hash % 360;
  const h2 = (h1 + 55 + (hash % 40)) % 360;
  return `linear-gradient(135deg, hsl(${h1} 55% 28%), hsl(${h2} 55% 16%))`;
};

// ── best-effort poster lookup (Wikipedia REST API, keyless, CORS-enabled) ──
const POSTER_CACHE_KEY = "p4_poster_cache_v1";
function loadPosterCache() {
  try {
    return JSON.parse(sessionStorage.getItem(POSTER_CACHE_KEY) || "{}");
  } catch (e) {
    return {};
  }
}
function savePosterCache(cache) {
  try {
    sessionStorage.setItem(POSTER_CACHE_KEY, JSON.stringify(cache));
  } catch (e) {
    /* storage unavailable (private mode, quota) — degrade silently */
  }
}
const posterCache = loadPosterCache();

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

P4.getPosterUrl = async function (movieId, title, year) {
  const cacheKey = String(movieId);
  if (cacheKey in posterCache) return posterCache[cacheKey];

  try {
    const query = encodeURIComponent(`${title} ${year || ""} film`);
    const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&list=search&srlimit=1&format=json&origin=*&srsearch=${query}`;
    const searchRes = await withTimeout(fetch(searchUrl), 2500);
    const searchJson = await searchRes.json();
    const hit = searchJson?.query?.search?.[0];
    if (!hit) {
      posterCache[cacheKey] = null;
      savePosterCache(posterCache);
      return null;
    }
    const pageTitle = encodeURIComponent(hit.title.replace(/ /g, "_"));
    const summaryUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${pageTitle}`;
    const summaryRes = await withTimeout(fetch(summaryUrl), 2500);
    const summaryJson = await summaryRes.json();
    const url = summaryJson?.thumbnail?.source || null;
    posterCache[cacheKey] = url;
    savePosterCache(posterCache);
    return url;
  } catch (e) {
    return null; // offline, blocked, or no match — fall back to placeholder
  }
};

// ── movie card rendering ─────────────────────────────────────────────────
P4.renderMovieCard = function (movie, { onClick, isStatic } = {}) {
  const card = document.createElement(onClick ? "button" : "div");
  card.type = onClick ? "button" : undefined;
  card.className = "movie-card" + (isStatic ? " static" : "");
  card.dataset.movieId = movie.id;

  const poster = document.createElement("div");
  poster.className = "movie-poster";
  poster.style.background = P4.gradientForTitle(movie.title);
  if (movie.score) {
    const score = document.createElement("span");
    score.className = "movie-score";
    score.textContent = `★ ${movie.score}`;
    poster.appendChild(score);
  }
  card.appendChild(poster);

  const body = document.createElement("div");
  body.className = "movie-body";

  const title = document.createElement("div");
  title.className = "movie-title";
  title.innerHTML = `${escapeHtml(movie.title)} <span class="movie-year">${movie.year || ""}</span>`;
  body.appendChild(title);

  if (movie.genres && movie.genres.length) {
    const genres = document.createElement("div");
    genres.className = "movie-genres";
    movie.genres.forEach((g) => {
      const badge = document.createElement("span");
      badge.className = "badge badge-secondary";
      badge.textContent = g;
      genres.appendChild(badge);
    });
    body.appendChild(genres);
  }

  if (movie.keywords && movie.keywords.length) {
    const kw = document.createElement("div");
    kw.className = "movie-keywords";
    kw.textContent = movie.keywords.join(" · ");
    body.appendChild(kw);
  }

  card.appendChild(body);

  if (onClick) {
    card.addEventListener("click", () => onClick(movie));
  }

  // best-effort poster swap-in
  P4.getPosterUrl(movie.id, movie.title, movie.year).then((url) => {
    if (url) {
      poster.style.backgroundImage = `url("${url}")`;
      poster.style.backgroundColor = "#111";
    }
  });

  return card;
};

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ── CSRF-free JSON post helper (submit endpoints are csrf_exempt) ───────
async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// ── Pairwise (and validation, same interaction) trial flow ──────────────
P4.PairwiseFlow = class {
  constructor(root, initialTrial, submitUrl, { onDone }) {
    this.root = root;
    this.submitUrl = submitUrl;
    this.onDone = onDone;
    this.render(initialTrial);
  }

  render(trial) {
    this.trial = trial;
    this.locked = false;
    this.startedAt = Date.now();

    this.root.querySelector(".progress-fill").style.width =
      `${((trial.trial_index) / Math.max(trial.total, 1)) * 100}%`;
    const label = this.root.querySelector(".progress-label-current");
    if (trial.phase === "practice") {
      label.textContent = "Practice round";
    } else {
      label.textContent = `Trial ${trial.trial_index + 1} of ${trial.total}`;
    }

    const banner = this.root.querySelector(".practice-banner");
    if (trial.phase === "practice") {
      const target = trial.movies.find((m) => m.id === trial.target_id);
      banner.style.display = "flex";
      banner.textContent = `Practice: please click "${target ? target.title : "the second movie"}" to continue.`;
    } else {
      banner.style.display = "none";
    }

    const grid = this.root.querySelector(".movie-grid-2");
    grid.innerHTML = "";
    const [a, b] = trial.movies;
    grid.appendChild(P4.renderMovieCard(a, { onClick: (m) => this.choose(m) }));
    const vs = document.createElement("div");
    vs.className = "movie-vs";
    vs.textContent = "OR";
    grid.appendChild(vs);
    grid.appendChild(P4.renderMovieCard(b, { onClick: (m) => this.choose(m) }));
  }

  async choose(movie) {
    if (this.locked) return;
    this.locked = true;

    this.root.querySelectorAll(".movie-card").forEach((card) => {
      card.classList.add(String(card.dataset.movieId) === String(movie.id) ? "chosen" : "rejected");
    });

    const response_time_ms = Date.now() - this.startedAt;
    const result = await postJson(this.submitUrl, { chosen_id: movie.id, response_time_ms });

    await new Promise((r) => setTimeout(r, 350));

    if (result.done) {
      this.onDone(result.redirect);
    } else {
      this.render(result.next);
    }
  }
};

// ── Ranking trial flow (click-to-build-order, no drag-and-drop needed) ──
P4.RankingFlow = class {
  constructor(root, initialTrial, submitUrl, { onDone }) {
    this.root = root;
    this.submitUrl = submitUrl;
    this.onDone = onDone;
    this.start(initialTrial);
  }

  start(trial) {
    this.trial = trial;
    this.movies = trial.movies;
    this.ranked = [];
    this.startedAt = Date.now();
    this.locked = false;
    this.renderProgress();
    this.renderBanner();
    this.renderBoard();
  }

  renderProgress() {
    this.root.querySelector(".progress-fill").style.width =
      `${(this.trial.trial_index / Math.max(this.trial.total, 1)) * 100}%`;
    const label = this.root.querySelector(".progress-label-current");
    label.textContent = this.trial.phase === "practice"
      ? "Practice round"
      : `Ranking ${this.trial.trial_index + 1} of ${this.trial.total}`;
  }

  renderBanner() {
    const banner = this.root.querySelector(".practice-banner");
    if (this.trial.phase === "practice") {
      const target = this.movies.find((m) => m.id === this.trial.target_id);
      banner.style.display = "flex";
      banner.textContent = `Practice: please place "${target ? target.title : "this movie"}" first (rank 1).`;
    } else {
      banner.style.display = "none";
    }
  }

  renderBoard() {
    const pool = this.root.querySelector(".ranking-pool");
    const slots = this.root.querySelector(".ranking-slots");
    pool.innerHTML = "";
    slots.innerHTML = "";

    this.movies.forEach((movie) => {
      const used = this.ranked.includes(movie.id);
      const card = document.createElement("button");
      card.type = "button";
      card.className = "ranking-mini-card" + (used ? " used" : "");
      card.innerHTML = `<span class="ranking-mini-title">${escapeHtml(movie.title)}</span>` +
        `<span class="ranking-mini-sub">${escapeHtml((movie.genres || []).slice(0, 2).join(", "))}${movie.year ? " · " + movie.year : ""}</span>`;
      if (!used) {
        card.addEventListener("click", () => this.place(movie.id));
      }
      pool.appendChild(card);
    });

    for (let i = 0; i < this.movies.length; i++) {
      const slot = document.createElement("div");
      const movieId = this.ranked[i];
      const movie = movieId != null ? this.movies.find((m) => m.id === movieId) : null;
      slot.className = "rank-slot" + (movie ? " filled" : "");
      if (movie) {
        slot.innerHTML = `<span class="rank-number">${i + 1}</span>` +
          `<span class="rank-slot-title">${escapeHtml(movie.title)}</span>` +
          `<span class="rank-slot-remove">✕ remove</span>`;
        slot.addEventListener("click", () => this.remove(movieId));
      } else {
        slot.innerHTML = `<span class="rank-number">${i + 1}</span>` +
          `<span class="rank-slot-hint">${i === 0 ? "most preferred" : i === this.movies.length - 1 ? "least preferred" : ""}</span>`;
      }
      slots.appendChild(slot);
    }

    if (this.ranked.length === this.movies.length) {
      this.submit();
    }
  }

  place(movieId) {
    if (this.locked || this.ranked.includes(movieId)) return;
    this.ranked.push(movieId);
    this.renderBoard();
  }

  remove(movieId) {
    if (this.locked) return;
    this.ranked = this.ranked.filter((id) => id !== movieId);
    this.renderBoard();
  }

  async submit() {
    this.locked = true;
    const response_time_ms = Date.now() - this.startedAt;
    const result = await postJson(this.submitUrl, { order: this.ranked, response_time_ms });
    if (result.done) {
      this.onDone(result.redirect);
    } else {
      this.start(result.next);
    }
  }
};

window.P4 = P4;
