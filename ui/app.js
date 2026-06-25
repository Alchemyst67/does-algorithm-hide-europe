(function () {
  const data = window.AUDIT_DATA;

  const metricOptions = [
    "NDCG@20",
    "Recall@20",
    "MAP@20",
    "Coverage@20",
    "European Exposure@20",
    "Non-English Exposure@20",
    "Long-tail Exposure@20",
    "PACPG European",
    "PACPG Non-English",
    "PACPG Long-tail"
  ];

  const cvMetricOptions = [
    "NDCG@20 mean",
    "NDCG@20 std",
    "Recall@20 mean",
    "MAP@20 mean",
    "Coverage@20 mean",
    "European Exposure@20 mean",
    "Non-English Exposure@20 mean",
    "Absolute PACPG mean"
  ];

  const feedbackMetricOptions = [
    "recommendation_european_share",
    "recommendation_non_english_share",
    "recommendation_us_origin_share",
    "recommendation_us_company_share",
    "recommendation_european_shift",
    "recommendation_non_english_shift",
    "recommendation_us_origin_shift",
    "recommendation_origin_jsd",
    "recommendation_language_jsd",
    "recommendation_popularity_jsd"
  ];

  const colors = {
    blue: "#2f6b9a",
    teal: "#167f86",
    green: "#2e8b57",
    gold: "#c99700",
    red: "#b84a4a",
    violet: "#6e5aa8",
    ink: "#142232",
    muted: "#5d6875",
    line: "#d5d1c7"
  };

  const fmt = new Intl.NumberFormat("en-US");
  const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
  const dec = (value, digits = 4) => Number(value).toFixed(digits);
  const byId = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[ch]));

  function runCount(name) {
    const row = data.runReport.find((item) => item.object === name);
    return row ? Number(row.count) : 0;
  }

  function metricValue(row, metric) {
    return Number(row[metric] ?? 0);
  }

  function isPercentMetric(metric) {
    return metric.includes("Coverage") || metric.includes("Exposure") || metric.includes("Recall") || metric.includes("NDCG") || metric.includes("MAP");
  }

  function formatMetric(metric, value) {
    if (metric.includes("PACPG") || metric.includes("ProminenceGap")) return dec(value, 4);
    if (metric.includes("jsd")) return dec(value, 3);
    if (metric.includes("share") || metric.includes("shift")) return pct(value);
    return isPercentMetric(metric) ? pct(value) : dec(value, 4);
  }

  function modelOrder(row) {
    return data.models.findIndex((item) => item.Model === row.Model);
  }

  function sortedModels(metric, sortMode) {
    const rows = [...data.models];
    if (sortMode === "original") return rows;
    rows.sort((a, b) => {
      const delta = metricValue(a, metric) - metricValue(b, metric);
      return sortMode === "asc" ? delta : -delta;
    });
    return rows;
  }

  function svgEl(tag, attrs = {}, children = []) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    children.forEach((child) => el.appendChild(child));
    return el;
  }

  function textEl(text, attrs = {}) {
    const el = svgEl("text", attrs);
    el.textContent = text;
    return el;
  }

  function renderHero() {
    byId("runScope").textContent = data.scope.run;
    const metrics = [
      ...data.datasetScale,
      { label: "Full-run users", value: runCount("full analysis users"), display: fmt.format(runCount("full analysis users")) },
      { label: "Candidate items", value: runCount("candidate items"), display: fmt.format(runCount("candidate items")) }
    ];
    byId("heroMetrics").innerHTML = metrics.map((item, index) => `
      <div class="metric-tile" style="border-left-color: ${[colors.teal, colors.gold, colors.green, colors.red, colors.blue, colors.violet][index % 6]}">
        <div class="metric-value">${esc(item.display)}</div>
        <div class="metric-label">${esc(item.label)}</div>
      </div>
    `).join("");
  }

  function findRow(rows, key, value) {
    return (rows || []).find((row) => row[key] === value);
  }

  function bestBy(rows, metric) {
    return [...(rows || [])].sort((a, b) => Number(b[metric] || 0) - Number(a[metric] || 0))[0];
  }

  function worstBy(rows, metric) {
    return [...(rows || [])].sort((a, b) => Number(a[metric] || 0) - Number(b[metric] || 0))[0];
  }

  function renderStoryline() {
    const movies = summaryValue("movies");
    const m3lItems = summaryValue("movies in M3L interaction item universe");
    const wikidata = summaryValue("movies with Wikidata match");
    const textCoverage = Number(findRow(data.moviesDbCoverage, "field", "has_text_mpnet_json")?.coverage_rate || 0);
    const imageCoverage = Number(findRow(data.moviesDbCoverage, "field", "has_image_clip_image_json")?.coverage_rate || 0);
    const fullUsers = runCount("full analysis users");
    const candidateItems = runCount("candidate items");

    const bestNdcg = bestBy(data.models, "NDCG@20");
    const worstEurope = worstBy(data.models, "European Exposure@20");
    const bestEurope = bestBy(data.models, "European Exposure@20");
    const mpnet = findRow(data.models, "Model", "MPNet-content");
    const clip = findRow(data.models, "Model", "CLIP-image-content");
    const hybrid = findRow(data.models, "Model", "Hybrid");
    const feedbackEuropeDrop = worstBy(data.feedbackFinalSummary, "recommendation_european_shift");
    const feedbackLanguageDrop = worstBy(data.feedbackFinalSummary, "recommendation_non_english_shift");
    const feedbackOriginRisk = bestBy(data.feedbackFinalSummary, "recommendation_origin_jsd");
    const rerankBest = bestBy(data.rerank, "European Exposure@20");

    const storyCards = [
      {
        label: "1. Data reality before modelling",
        title: `${fmt.format(movies)} catalogue movies, ${pct(m3lItems / movies)} in the M3L item universe`,
        body: `The dashboard starts from files, joins and coverage. MPNet JSON coverage is ${pct(textCoverage)} and CLIP-image JSON coverage is ${pct(imageCoverage)}; missing fields remain visible in the ledgers.`
      },
      {
        label: "2. Catalogue is not visibility",
        title: `${worstEurope.Model} gives only ${pct(worstEurope["European Exposure@20"])} European Top-K exposure`,
        body: `${bestEurope.Model} reaches ${pct(bestEurope["European Exposure@20"])} European exposure, so ranking visibility changes materially by model even before mitigation.`
      },
      {
        label: "3. Accuracy alone is insufficient",
        title: `${bestNdcg.Model} leads NDCG@20 at ${dec(bestNdcg["NDCG@20"], 4)}`,
        body: `The same table also reports exposure and PACPG, because high utility does not by itself answer whether European, non-English or long-tail films become visible.`
      },
      {
        label: "4. Multimodal features shift the hierarchy",
        title: `MPNet and CLIP do not behave like the collaborative baselines`,
        body: `MPNet-content has ${pct(mpnet["Non-English Exposure@20"])} non-English exposure; CLIP-image-content has ${pct(clip["European Exposure@20"])} European exposure. The question is not only better/worse, but which kind of Europe becomes legible to each modality.`
      },
      {
        label: "5. Feedback loops make drift observable",
        title: `${feedbackEuropeDrop.Model} has the largest European-origin recommendation drop`,
        body: `In the Schedl-style loop, ${feedbackEuropeDrop.Model} shifts ${pct(feedbackEuropeDrop.recommendation_european_shift)} against initial histories; ${feedbackLanguageDrop.Model} has the largest non-English drop; ${feedbackOriginRisk.Model} has the highest origin JSD.`
      },
      {
        label: "6. Mitigation is a transparent trade-off",
        title: `${rerankBest.Model} reaches ${pct(rerankBest["European Exposure@20"])} European exposure`,
        body: `The lambda sweep reports NDCG@20, Recall@20, exposure and PACPG together, so governance pressure is explicit rather than hidden inside the model.`
      }
    ];

    byId("storyCards").innerHTML = storyCards.map((card) => `
      <article class="story-card">
        <span>${esc(card.label)}</span>
        <h3>${esc(card.title)}</h3>
        <p>${esc(card.body)}</p>
      </article>
    `).join("");

    const evidenceStatus = [
      ["Aggregate Europe", "Supported", "Model comparison, CV and feedback-loop outputs already report European, non-English and long-tail visibility."],
      ["Which Europe", "Partial", "The current UI has the storyline and caveats, but no full country/region exposure tables yet. No granular country claim is made here."],
      ["Languages", "Partial", "Aggregate non-English is supported; fine-grained language hierarchy needs a dedicated language exposure table."],
      ["Spain case study", "Not yet claimed", "Spain-origin, Spanish-language and European Spanish-language definitions are conceptually separated, but reliable output tables are not yet present."],
      ["Compatibility mechanism", "Partial", "Popularity, language, US-origin/company and co-production caveats exist; matched-pair and regression mechanism outputs are still a next analysis layer."],
      ["Religion/theme", "Insufficient evidence", "No main fairness claim is made. Film-theme analysis would need reliable metadata support counts first."]
    ];
    byId("evidenceStatusCards").innerHTML = evidenceStatus.map(([name, status, note]) => `
      <article class="status-card status-${esc(status.toLowerCase().replaceAll(" ", "-"))}">
        <div>
          <h3>${esc(name)}</h3>
          <span>${esc(status)}</span>
        </div>
        <p>${esc(note)}</p>
      </article>
    `).join("");

    const funnelRows = [
      ["Catalogue", fmt.format(movies), "All MovieLens movies in the combined movie-level table."],
      ["M3L item universe", fmt.format(m3lItems), "Items available for interaction-based recommendation."],
      ["Wikidata matched", wikidata ? fmt.format(wikidata) : "reported in coverage ledger", "Country/language metadata enters through cached Wikidata joins."],
      ["Full-run sample", fmt.format(fullUsers), "Users used for the local full-run recommender analysis."],
      ["Candidate items", fmt.format(candidateItems), "Items scored in the full-run candidate universe."],
      ["Top-K visibility", "K = 20", "Final visibility is rank-discounted exposure in recommendation lists."],
      ["Feedback loop", `${fmt.format(Number((data.feedbackRunReport || []).find((row) => row.object === "feedback-loop users")?.count || 0))} users`, "Repeated recommendation and simulated consumption expose dynamic drift."]
    ];
    byId("visibilityFunnel").innerHTML = funnelRows.map(([label, value, note], index) => `
      <article class="funnel-card">
        <span>${index + 1}</span>
        <h3>${esc(label)}</h3>
        <strong>${esc(value)}</strong>
        <p>${esc(note)}</p>
      </article>
    `).join("");
    byId("funnelInterpretation").textContent =
      `The governance finding is read across the funnel: ${fmt.format(movies)} catalogue movies do not automatically become ranked attention. ` +
      `Model outputs, cross-validation and feedback-loop drift show which parts of the catalogue become visible, repeatedly visible or still under-evidenced.`;
  }

  function renderDatasetFoundation() {
    const exploration = data.exploration;
    if (!exploration) {
      byId("dataset-foundation").hidden = true;
      return;
    }

    // This section treats Nico's exploration notebook as provenance evidence, not as a model result.
    byId("foundationKpis").innerHTML = exploration.kpis.map((item) => `
      <div class="foundation-kpi">
        <div class="foundation-kpi-value">${esc(item.display)}</div>
        <div class="foundation-kpi-label">${esc(item.label)}</div>
        <p>${esc(item.note)}</p>
      </div>
    `).join("");

    const tabEntries = Object.entries(exploration.tabs);
    byId("foundationTabs").innerHTML = tabEntries.map(([key, tab], index) => `
      <button class="segment-button ${index === 0 ? "active" : ""}" data-foundation-tab="${esc(key)}" role="tab">
        ${esc(tab.label)}
      </button>
    `).join("");

    byId("explorationFigureSelect").innerHTML = exploration.figures.map((figure, index) => `
      <option value="${index}">${esc(figure.title)}</option>
    `).join("");

    document.querySelectorAll("[data-foundation-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-foundation-tab]").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        updateFoundationTab(button.dataset.foundationTab);
      });
    });
    byId("explorationFigureSelect").addEventListener("change", () => {
      renderExplorationFigure(Number(byId("explorationFigureSelect").value));
    });

    updateFoundationTab(tabEntries[0][0]);
    renderExplorationFigure(0);
  }

  function summaryValue(metricName) {
    const row = (data.moviesDbSummary || []).find((item) => item.metric === metricName);
    return row ? Number(row.value) : 0;
  }

  function renderMoviesDb() {
    if (!data.moviesDbSummary?.length) {
      byId("movies-db").hidden = true;
      return;
    }
    const movies = summaryValue("movies");
    const rated = summaryValue("movies with ratings in loaded sample");
    const tags = summaryValue("movies with user tags");
    const m3lItems = summaryValue("movies in M3L interaction item universe");
    const text = summaryValue("movies with text JSON feature");
    const image = summaryValue("movies with image JSON feature");
    const wikidata = summaryValue("movies with Wikidata match");

    const kpis = [
      ["Movies", movies, fmt.format(movies), "Rows in the combined movie-level table."],
      ["Rated sample", rated, fmt.format(rated), "Movies with at least one loaded rating."],
      ["User tags", tags, fmt.format(tags), "Movies with explicit user tag metadata."],
      ["M3L items", m3lItems, fmt.format(m3lItems), "MovieLens IDs mapped from the M3L interaction universe."],
      ["MPNet JSON", text, pct(text / movies), "MovieLens movies with MPNet JSON feature files."],
      ["CLIP image", image, pct(image / movies), "MovieLens movies with CLIP-image JSON feature files."],
      ["Wikidata", wikidata, pct(wikidata / movies), "Movies matched to cached Wikidata metadata."]
    ];
    byId("moviesDbKpis").innerHTML = kpis.map(([label, value, display, note]) => `
      <div class="foundation-kpi">
        <div class="foundation-kpi-value">${esc(display)}</div>
        <div class="foundation-kpi-label">${esc(label)}</div>
        <p>${esc(note)}</p>
      </div>
    `).join("");

    const missing = (data.moviesDbInventory || []).filter((row) => !row.exists || row.is_lfs_pointer);
    byId("moviesDbInterpretation").textContent =
      `The current Movies DB contains ${fmt.format(movies)} MovieLens movies. It combines ratings, tags, genome tags, feature coverage and cached Wikidata enrichment into one row per movie. ` +
      "This is the catalogue layer for the later cultural-prominence audit.";
    byId("moviesDbLocalNote").textContent =
      `${missing.length} expected files from Nico's raw M3L folder are not present in this local project directory, mainly raw plot/poster/trailer TSVs and M3L-10M files. ` +
      "The pipeline reports them as missing and keeps the current table real-data only.";

    const overlap = data.moviesDbOverlap || [];
    byId("moviesDbOverlap").innerHTML = overlap.map((row) => `
      <div><strong>${esc(row.comparison)}:</strong> ${esc(fmt.format(Number(row.intersection || 0)))} matched, Jaccard ${esc(pct(Number(row.jaccard || 0)))}</div>
    `).join("");

    const coverageRows = (data.moviesDbCoverage || [])
      .filter((row) => ["has_text_mpnet_json", "has_image_clip_image_json", "in_m3l_interaction_items", "has_wikidata_match", "has_country", "has_original_language"].includes(row.field))
      .map((row) => ({ Model: row.field.replaceAll("_", " "), "Coverage": Number(row.coverage_rate) }));
    renderBarChart(byId("moviesDbCoverageChart"), coverageRows, "Coverage");

    const figures = [
      ["Coverage", data.assets.moviesDbCoverage, "Coverage by source and feature family."],
      ["Ratings", data.assets.moviesDbRatingDistribution, "Rating means and rating-count skew."],
      ["Genre visibility leads", data.assets.moviesDbGenreVisibility, "Catalogue, interest and baseline Top-K shares."],
      ["User concentration", data.assets.moviesDbUserConcentration, "Heavy-user contribution to loaded ratings."]
    ];
    byId("moviesDbFigureGrid").innerHTML = figures.map(([title, src, caption]) => `
      <button class="evidence-item" data-src="${esc(src)}" data-title="${esc(title)}" data-caption="${esc(caption)}">
        <img src="${esc(src)}" alt="${esc(title)}" />
        <h3>${esc(title)}</h3>
        <p>${esc(caption)}</p>
      </button>
    `).join("");
  }

  function updateFoundationTab(tabKey) {
    const tab = data.exploration.tabs[tabKey];
    byId("foundationTitle").textContent = tab.label;
    byId("foundationClaim").textContent = tab.claim;
    byId("foundationEvidence").innerHTML = tab.evidence.map((item) => `<li>${esc(item)}</li>`).join("");
    byId("foundationMethod").textContent = tab.methodologicalImplication;
  }

  function renderExplorationFigure(index) {
    const figure = data.exploration.figures[index] || data.exploration.figures[0];
    byId("explorationFigure").src = figure.path;
    byId("explorationFigure").alt = figure.title;
    byId("explorationCaption").textContent = `${figure.title}. ${figure.interpretation}`;
  }

  function renderBarChart(host, rows, metric) {
    const width = 860;
    const height = 330;
    const margin = { top: 34, right: 22, bottom: 92, left: 62 };
    const values = rows.map((row) => metricValue(row, metric));
    const minValue = Math.min(0, ...values);
    const maxValue = Math.max(...values, metric.includes("PACPG") ? 0 : 0.001);
    const span = maxValue - minValue || 1;
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;
    const barW = chartW / rows.length * 0.68;
    const zeroY = margin.top + chartH - ((0 - minValue) / span) * chartH;
    const palette = [colors.blue, colors.teal, colors.green, colors.gold, colors.red, colors.violet, colors.ink, colors.muted];

    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%" });
    svg.appendChild(textEl(metric, { x: margin.left, y: 18, class: "chart-title" }));
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY, stroke: colors.line, "stroke-width": 1 }));

    rows.forEach((row, index) => {
      const x = margin.left + index * (chartW / rows.length) + (chartW / rows.length - barW) / 2;
      const value = metricValue(row, metric);
      const y = margin.top + chartH - ((value - minValue) / span) * chartH;
      const h = Math.abs(zeroY - y);
      const rectY = value >= 0 ? y : zeroY;
      svg.appendChild(svgEl("rect", { x, y: rectY, width: barW, height: Math.max(1, h), fill: palette[index % palette.length] }));
      svg.appendChild(textEl(formatMetric(metric, value), {
        x: x + barW / 2,
        y: value >= 0 ? y - 6 : zeroY + h + 14,
        "text-anchor": "middle",
        class: "tick-label"
      }));
      const label = row.Model.replace("Hybrid + reranking ", "Rerank ");
      const text = textEl(label, {
        x: x + barW / 2,
        y: height - 70,
        "text-anchor": "end",
        transform: `rotate(-32 ${x + barW / 2} ${height - 70})`,
        class: "tick-label"
      });
      svg.appendChild(text);
    });

    const yLabels = [minValue, 0, maxValue];
    yLabels.forEach((value) => {
      const y = margin.top + chartH - ((value - minValue) / span) * chartH;
      svg.appendChild(textEl(formatMetric(metric, value), { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "tick-label" }));
      svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: colors.line, "stroke-width": 0.6, opacity: 0.55 }));
    });

    host.innerHTML = "";
    host.appendChild(svg);
  }

  function renderModelTable(rows) {
    const cols = ["Model", "NDCG@20", "Recall@20", "Coverage@20", "European Exposure@20", "Non-English Exposure@20", "Long-tail Exposure@20", "PACPG European", "PACPG Non-English", "PACPG Long-tail"];
    byId("modelTable").innerHTML = `
      <thead><tr>${cols.map((col) => `<th>${esc(col)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${cols.map((col) => {
            const value = col === "Model" ? row[col] : formatMetric(col, row[col]);
            return `<td>${esc(value)}</td>`;
          }).join("")}</tr>
        `).join("")}
      </tbody>
    `;
  }

  function cvRows() {
    return (data.cvModelSummary || []).map((row) => ({
      ...row,
      "Absolute PACPG mean": Math.abs(Number(row["PACPG European mean"] || 0)) +
        Math.abs(Number(row["PACPG Non-English mean"] || 0)) +
        Math.abs(Number(row["PACPG Long-tail mean"] || 0))
    }));
  }

  function renderCvTable(rows) {
    const cols = ["Model", "NDCG@20 mean", "NDCG@20 std", "Recall@20 mean", "Coverage@20 mean", "European Exposure@20 mean", "Non-English Exposure@20 mean", "Absolute PACPG mean"];
    byId("cvTable").innerHTML = `
      <thead><tr>${cols.map((col) => `<th>${esc(col)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${cols.map((col) => `<td>${esc(col === "Model" ? row[col] : formatMetric(col, row[col]))}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    `;
  }

  function renderCvSection() {
    const rows = cvRows();
    if (!rows.length) {
      byId("cross-validation").hidden = true;
      return;
    }
    const metric = byId("cvMetricSelect").value || cvMetricOptions[0];
    const sortMode = metric === "Absolute PACPG mean" || metric.includes("std") ? "asc" : "desc";
    const sorted = [...rows].sort((a, b) => sortMode === "asc" ? Number(a[metric]) - Number(b[metric]) : Number(b[metric]) - Number(a[metric]));
    renderBarChart(byId("cvChart"), sorted, metric);
    renderCvTable(rows);

    const best = sorted[0];
    const utilityBest = [...rows].sort((a, b) => Number(b["NDCG@20 mean"]) - Number(a["NDCG@20 mean"]))[0];
    const pacpgBest = [...rows].sort((a, b) => Number(a["Absolute PACPG mean"]) - Number(b["Absolute PACPG mean"]))[0];
    byId("cvInterpretation").textContent =
      `For ${metric}, the leading model is ${best.Model} (${formatMetric(metric, best[metric])}). ` +
      `Across the robustness check, ${utilityBest.Model} has the strongest mean NDCG@20, while ${pacpgBest.Model} has the lowest mean absolute PACPG. ` +
      "This checks sample stability across user folds, not a new data source.";

    const report = Object.fromEntries((data.cvRunReport || []).map((row) => [row.setting, row.value]));
    byId("cvProtocol").innerHTML = `
      <div><strong>Folds:</strong> ${esc(report.folds)}</div>
      <div><strong>Sampled users:</strong> ${esc(fmt.format(Number(report["sampled users"] || 0)))}</div>
      <div><strong>Users per fold:</strong> ${esc(fmt.format(Number(report["users per fold"] || 0)))}</div>
      <div><strong>Candidate pool:</strong> top ${esc(fmt.format(Number(report["top popular candidate items"] || 0)))} + fold items</div>
    `;
  }

  function renderLineChart(host, rows, metric) {
    const width = 860;
    const height = 350;
    const margin = { top: 36, right: 28, bottom: 46, left: 72 };
    const models = [...new Set(rows.map((row) => row.Model))];
    const iterations = [...new Set(rows.map((row) => Number(row.iteration)))].sort((a, b) => a - b);
    const values = rows.map((row) => Number(row[metric] || 0));
    const minValue = Math.min(0, ...values);
    const maxValue = Math.max(...values, 0.001);
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;
    const xScale = (value) => margin.left + ((value - iterations[0]) / ((iterations.at(-1) - iterations[0]) || 1)) * chartW;
    const yScale = (value) => margin.top + (1 - ((value - minValue) / ((maxValue - minValue) || 1))) * chartH;
    const palette = [colors.blue, colors.teal, colors.green, colors.gold, colors.red, colors.violet];
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%" });

    svg.appendChild(textEl(metric.replaceAll("_", " "), { x: margin.left, y: 18, class: "chart-title" }));
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yScale(0), y2: yScale(0), stroke: colors.line, "stroke-width": 1 }));
    svg.appendChild(svgEl("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, stroke: colors.line, "stroke-width": 1 }));

    models.forEach((model, index) => {
      const modelRows = rows.filter((row) => row.Model === model).sort((a, b) => Number(a.iteration) - Number(b.iteration));
      const points = modelRows.map((row) => `${xScale(Number(row.iteration))},${yScale(Number(row[metric] || 0))}`).join(" ");
      svg.appendChild(svgEl("polyline", { points, fill: "none", stroke: palette[index % palette.length], "stroke-width": 2.2 }));
      modelRows.forEach((row) => {
        svg.appendChild(svgEl("circle", {
          cx: xScale(Number(row.iteration)),
          cy: yScale(Number(row[metric] || 0)),
          r: 3.5,
          fill: palette[index % palette.length]
        }));
      });
      const last = modelRows.at(-1);
      svg.appendChild(textEl(model, {
        x: xScale(Number(last.iteration)) + 8,
        y: yScale(Number(last[metric] || 0)) + 4,
        class: "tick-label"
      }));
    });

    iterations.forEach((value) => {
      svg.appendChild(textEl(String(value), { x: xScale(value), y: height - 18, "text-anchor": "middle", class: "tick-label" }));
    });
    [minValue, 0, maxValue].forEach((value) => {
      const y = yScale(value);
      svg.appendChild(textEl(formatMetric(metric, value), { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "tick-label" }));
      svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, stroke: colors.line, "stroke-width": 0.6, opacity: 0.5 }));
    });
    svg.appendChild(textEl("Iteration", { x: width - margin.right - 52, y: height - 18, class: "axis-label" }));

    host.innerHTML = "";
    host.appendChild(svg);
  }

  function renderFeedbackTable(rows) {
    const cols = [
      "Model",
      "recommendation_european_shift",
      "recommendation_non_english_shift",
      "recommendation_us_origin_shift",
      "recommendation_us_company_shift",
      "recommendation_origin_jsd",
      "recommendation_language_jsd",
      "recommendation_popularity_jsd"
    ];
    byId("feedbackTable").innerHTML = `
      <thead><tr>${cols.map((col) => `<th>${esc(col.replaceAll("_", " "))}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${cols.map((col) => `<td>${esc(col === "Model" ? row[col] : formatMetric(col, row[col]))}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    `;
  }

  function renderFeedbackLoop() {
    const iterationRows = data.feedbackIterationMetrics || [];
    const finalRows = data.feedbackFinalSummary || [];
    if (!iterationRows.length || !finalRows.length) {
      byId("feedback-loop").hidden = true;
      return;
    }
    const metric = byId("feedbackMetricSelect").value || feedbackMetricOptions[0];
    renderLineChart(byId("feedbackChart"), iterationRows, metric);

    const sortedByOrigin = [...finalRows].sort((a, b) => Number(b.recommendation_origin_jsd) - Number(a.recommendation_origin_jsd));
    const europeDrop = [...finalRows].sort((a, b) => Number(a.recommendation_european_shift) - Number(b.recommendation_european_shift))[0];
    const languageDrop = [...finalRows].sort((a, b) => Number(a.recommendation_non_english_shift) - Number(b.recommendation_non_english_shift))[0];
    const usGain = [...finalRows].sort((a, b) => Number(b.recommendation_us_origin_shift) - Number(a.recommendation_us_origin_shift))[0];
    const originWorst = sortedByOrigin[0];

    byId("feedbackInterpretation").textContent =
      `The feedback-loop run adapts Schedl et al.'s country-representation simulation to MovieLens by using each user's initial history as the cultural calibration target. ` +
      `${europeDrop.Model} shows the largest European-origin drop (${formatMetric("recommendation_european_shift", europeDrop.recommendation_european_shift)}), ` +
      `${languageDrop.Model} shows the largest non-English drop (${formatMetric("recommendation_non_english_shift", languageDrop.recommendation_non_english_shift)}), ` +
      `and ${usGain.Model} shows the strongest US-origin gain (${formatMetric("recommendation_us_origin_shift", usGain.recommendation_us_origin_shift)}). ` +
      `The highest origin-distribution miscalibration is ${originWorst.Model} (JSD ${dec(originWorst.recommendation_origin_jsd, 3)}).`;

    const report = Object.fromEntries((data.feedbackRunReport || []).map((row) => [row.object, row.count]));
    byId("feedbackProtocol").innerHTML = `
      <div><strong>Users:</strong> ${esc(fmt.format(Number(report["feedback-loop users"] || 0)))}</div>
      <div><strong>Candidate items:</strong> ${esc(fmt.format(Number(report["candidate items"] || 0)))}</div>
      <div><strong>Iterations:</strong> ${esc(fmt.format(Number(report.iterations || 0)))}</div>
      <div><strong>Top-K:</strong> ${esc(fmt.format(Number(report["top-k recommendations"] || 0)))}</div>
      <div><strong>Acceptance alpha:</strong> ${esc(report["acceptance alpha"] ?? "")}</div>
    `;

    renderFeedbackTable(finalRows);

    const figures = [
      ["Dynamics", data.assets.feedbackDynamics, "European and non-English shares over feedback-loop iterations."],
      ["JSD heatmap", data.assets.feedbackJsd, "Origin, language and popularity miscalibration."],
      ["Final shift", data.assets.feedbackShift, "Final recommendation shifts against initial user histories."],
      ["Country/language panels", data.assets.feedbackLanguageCountry, "Final composition and country/language JSD."]
    ];
    byId("feedbackFigureGrid").innerHTML = figures.map(([title, src, caption]) => `
      <button class="evidence-item" data-src="${esc(src)}" data-title="${esc(title)}" data-caption="${esc(caption)}">
        <img src="${esc(src)}" alt="${esc(title)}" />
        <h3>${esc(title)}</h3>
        <p>${esc(caption)}</p>
      </button>
    `).join("");
    byId("feedbackFigureGrid").querySelectorAll(".evidence-item").forEach((button) => {
      button.addEventListener("click", () => {
        byId("dialogImage").src = button.dataset.src;
        byId("dialogImage").alt = button.dataset.title;
        byId("dialogCaption").textContent = `${button.dataset.title}: ${button.dataset.caption}`;
        byId("imageDialog").showModal();
      });
    });
  }

  function updateMetricInterpretation(rows, metric, group) {
    const best = [...data.models].sort((a, b) => metricValue(b, metric) - metricValue(a, metric))[0];
    const worst = [...data.models].sort((a, b) => metricValue(a, metric) - metricValue(b, metric))[0];
    const exposureMetric = `${group} Exposure@20`;
    const groupBest = [...data.models].sort((a, b) => metricValue(b, exposureMetric) - metricValue(a, exposureMetric))[0];
    const groupWorst = [...data.models].sort((a, b) => metricValue(a, exposureMetric) - metricValue(b, exposureMetric))[0];
    byId("metricInterpretation").textContent =
      `For ${metric}, ${best.Model} is highest at ${formatMetric(metric, best[metric])}, while ${worst.Model} is lowest at ${formatMetric(metric, worst[metric])}. ` +
      `For ${group} exposure, the range runs from ${groupWorst.Model} (${pct(groupWorst[exposureMetric])}) to ${groupBest.Model} (${pct(groupBest[exposureMetric])}). ` +
      "Read this as ranking visibility, not as a statement about catalogue availability.";
  }

  function renderPairComparison(metric) {
    const a = data.models.find((row) => row.Model === byId("compareA").value) || data.models[0];
    const b = data.models.find((row) => row.Model === byId("compareB").value) || data.models[1];
    const delta = metricValue(b, metric) - metricValue(a, metric);
    const pacpgA = Math.abs(a["PACPG European"]) + Math.abs(a["PACPG Non-English"]) + Math.abs(a["PACPG Long-tail"]);
    const pacpgB = Math.abs(b["PACPG European"]) + Math.abs(b["PACPG Non-English"]) + Math.abs(b["PACPG Long-tail"]);
    byId("pairComparison").innerHTML = `
      <strong>${esc(b.Model)}</strong> vs <strong>${esc(a.Model)}</strong><br>
      ${esc(metric)} delta: <strong>${delta >= 0 ? "+" : ""}${formatMetric(metric, delta)}</strong><br>
      Combined absolute PACPG delta: <strong>${(pacpgB - pacpgA) >= 0 ? "+" : ""}${dec(pacpgB - pacpgA, 4)}</strong>
    `;
  }

  function renderModelExplorer() {
    const metric = byId("metricSelect").value;
    const group = byId("groupSelect").value;
    const rows = sortedModels(metric, byId("sortSelect").value);
    renderBarChart(byId("modelChart"), rows, metric);
    renderModelTable(rows);
    updateMetricInterpretation(rows, metric, group);
    renderPairComparison(metric);
  }

  function renderRerankChart(selectedModel) {
    const width = 840;
    const height = 330;
    const margin = { top: 34, right: 28, bottom: 54, left: 68 };
    const rows = data.rerank.map((row) => {
      const absPacpg = Math.abs(row["PACPG European"]) + Math.abs(row["PACPG Non-English"]) + Math.abs(row["PACPG Long-tail"]);
      return { ...row, lambda: Number(row.Model.match(/lambda=([0-9.]+)/)?.[1] ?? 0), absPacpg };
    });
    const base = rows.find((row) => row.lambda === 0) || rows[0];
    rows.forEach((row) => {
      row.improvement = base.absPacpg - row.absPacpg;
    });
    const xVals = rows.map((row) => Number(row["NDCG@20"]));
    const yVals = rows.map((row) => row.improvement);
    const xMin = Math.min(...xVals) - 0.0003;
    const xMax = Math.max(...xVals) + 0.0003;
    const yMin = Math.min(...yVals, 0) - 0.001;
    const yMax = Math.max(...yVals, 0) + 0.001;
    const xScale = (value) => margin.left + ((value - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
    const yScale = (value) => margin.top + (1 - ((value - yMin) / (yMax - yMin || 1))) * (height - margin.top - margin.bottom);
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%" });
    svg.appendChild(textEl("Utility vs PACPG improvement", { x: margin.left, y: 18, class: "chart-title" }));
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: yScale(0), y2: yScale(0), stroke: colors.line, "stroke-width": 1 }));
    svg.appendChild(svgEl("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, stroke: colors.line, "stroke-width": 1 }));
    rows.forEach((row) => {
      const active = row.Model === selectedModel;
      svg.appendChild(svgEl("circle", {
        cx: xScale(row["NDCG@20"]),
        cy: yScale(row.improvement),
        r: active ? 8 : 5,
        fill: active ? colors.gold : colors.teal,
        stroke: colors.ink,
        "stroke-width": active ? 2 : 0
      }));
      svg.appendChild(textEl(`l=${row.lambda.toFixed(1)}`, {
        x: xScale(row["NDCG@20"]) + 8,
        y: yScale(row.improvement) - 8,
        class: "tick-label"
      }));
    });
    svg.appendChild(textEl("NDCG@20", { x: width - margin.right - 46, y: height - 16, class: "axis-label" }));
    svg.appendChild(textEl("PACPG improvement", { x: 12, y: margin.top + 12, class: "axis-label" }));
    byId("rerankChart").innerHTML = "";
    byId("rerankChart").appendChild(svg);
  }

  function renderLambda(selectedModel) {
    const row = data.rerank.find((item) => item.Model === selectedModel) || data.rerank[0];
    renderRerankChart(row.Model);
    const absPacpg = Math.abs(row["PACPG European"]) + Math.abs(row["PACPG Non-English"]) + Math.abs(row["PACPG Long-tail"]);
    byId("lambdaInterpretation").textContent =
      `${row.Model} reaches NDCG@20 ${dec(row["NDCG@20"], 4)} and combined absolute PACPG ${dec(absPacpg, 4)}. ` +
      "The selected point shows how much cultural-prominence pressure is applied after the hybrid ranking is already trained.";
    byId("lambdaMetrics").innerHTML = `
      <div><strong>NDCG@20:</strong> ${dec(row["NDCG@20"], 4)}</div>
      <div><strong>Recall@20:</strong> ${dec(row["Recall@20"], 4)}</div>
      <div><strong>European exposure:</strong> ${pct(row["European Exposure@20"])}</div>
      <div><strong>Non-English exposure:</strong> ${pct(row["Non-English Exposure@20"])}</div>
      <div><strong>Combined absolute PACPG:</strong> ${dec(absPacpg, 4)}</div>
    `;
  }

  function renderProxyChart() {
    const rows = [...data.proxyRisk].sort((a, b) => Number(b.share) - Number(a.share));
    const width = 860;
    const height = 350;
    const margin = { top: 36, right: 44, bottom: 36, left: 230 };
    const chartW = width - margin.left - margin.right;
    const rowH = (height - margin.top - margin.bottom) / rows.length;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%" });
    svg.appendChild(textEl("Metadata proxy-risk shares", { x: margin.left, y: 18, class: "chart-title" }));
    rows.forEach((row, index) => {
      const y = margin.top + index * rowH + 4;
      const w = Number(row.share) * chartW;
      const color = index < 2 ? colors.green : index < 5 ? colors.teal : colors.red;
      svg.appendChild(textEl(row.check, { x: margin.left - 12, y: y + rowH / 2 + 4, "text-anchor": "end", class: "tick-label" }));
      svg.appendChild(svgEl("rect", { x: margin.left, y, width: Math.max(2, w), height: rowH - 8, fill: color }));
      svg.appendChild(textEl(pct(row.share), { x: margin.left + w + 8, y: y + rowH / 2 + 4, class: "bar-label" }));
    });
    byId("proxyChart").innerHTML = "";
    byId("proxyChart").appendChild(svg);
  }

  function renderEvidence() {
    const coreItems = [
      ["Data pipeline", data.assets.pipeline, "How interactions, features, MovieLens and Wikidata form the audit dataset."],
      ["Join funnel", data.assets.joinFunnel, "Where metadata coverage shrinks before label construction."],
      ["Coverage", data.assets.coverage, "Feature and metadata coverage used to qualify the audit."],
      ["Long-tail distribution", data.assets.longTail, "Why popularity-sensitive evaluation matters."],
      ["Catalogue vs interaction share", data.assets.catalogueShare, "Availability and observed attention are not the same."],
      ["Research gap", data.assets.researchGap, "The contribution: cultural prominence under multimodal recommendation."],
      ["Group exposure", data.assets.groupExposure, "Model-level visibility differences by audit group."],
      ["Accuracy vs prominence", data.assets.accuracyFairness, "Trade-off view for utility and PACPG."],
      ["Cross-validation stability", data.assets.cvStability, "Fold-level stability for utility and PACPG."],
      ["Movies DB coverage", data.assets.moviesDbCoverage, "Movie-level coverage by source and feature family."],
      ["Movies DB ratings", data.assets.moviesDbRatingDistribution, "Rating means and rating-count skew in the loaded sample."],
      ["Movies DB genre leads", data.assets.moviesDbGenreVisibility, "Diagnostic genre-level audit leads."],
      ["Movies DB user concentration", data.assets.moviesDbUserConcentration, "Rating concentration among active users."],
      ["Feedback-loop dynamics", data.assets.feedbackDynamics, "Schedl-style representation dynamics across repeated recommendation."],
      ["Feedback-loop JSD", data.assets.feedbackJsd, "Origin, language and popularity miscalibration."],
      ["Feedback-loop final shift", data.assets.feedbackShift, "Final Top-K shifts relative to initial user histories."],
      ["Language/country feedback panels", data.assets.feedbackLanguageCountry, "Country and language bias after the feedback loop."],
      ["Workplan", data.assets.workplan, "Milestones for the final project."],
      ["Proxy-risk caveats", data.assets.proxyRisk, "Language, country and company-control checks."]
    ];
    const explorationItems = (data.exploration?.figures || []).map((figure) => [
      `Exploration: ${figure.title}`,
      figure.path,
      figure.interpretation
    ]);
    const items = [...coreItems, ...explorationItems];
    byId("evidenceGrid").innerHTML = items.map(([title, src, caption]) => `
      <button class="evidence-item" data-src="${esc(src)}" data-title="${esc(title)}" data-caption="${esc(caption)}">
        <img src="${esc(src)}" alt="${esc(title)}" />
        <h3>${esc(title)}</h3>
        <p>${esc(caption)}</p>
      </button>
    `).join("");
    document.querySelectorAll(".evidence-item").forEach((button) => {
      button.addEventListener("click", () => {
        byId("dialogImage").src = button.dataset.src;
        byId("dialogImage").alt = button.dataset.title;
        byId("dialogCaption").textContent = `${button.dataset.title}: ${button.dataset.caption}`;
        byId("imageDialog").showModal();
      });
    });
    byId("dialogClose").addEventListener("click", () => byId("imageDialog").close());
  }

  function renderSources() {
    const rows = data.citations;
    byId("sourceTable").innerHTML = `
      <thead><tr><th>Source</th><th>Used for</th><th>Citation</th><th>URL</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${esc(row.source)}</td>
            <td>${esc(row.used_for)}</td>
            <td>${esc(row.citation)}</td>
            <td><a href="${esc(row.url)}">${esc(row.url)}</a></td>
          </tr>
        `).join("")}
      </tbody>
    `;
  }

  function populateControls() {
    byId("metricSelect").innerHTML = metricOptions.map((metric) => `<option value="${esc(metric)}">${esc(metric)}</option>`).join("");
    byId("metricSelect").value = "NDCG@20";
    byId("cvMetricSelect").innerHTML = cvMetricOptions.map((metric) => `<option value="${esc(metric)}">${esc(metric)}</option>`).join("");
    byId("cvMetricSelect").value = "NDCG@20 mean";
    byId("feedbackMetricSelect").innerHTML = feedbackMetricOptions.map((metric) => `<option value="${esc(metric)}">${esc(metric.replaceAll("_", " "))}</option>`).join("");
    byId("feedbackMetricSelect").value = "recommendation_european_share";
    const modelOptions = data.models.map((row) => `<option value="${esc(row.Model)}">${esc(row.Model)}</option>`).join("");
    byId("compareA").innerHTML = modelOptions;
    byId("compareB").innerHTML = modelOptions;
    byId("compareA").value = "Hybrid";
    byId("compareB").value = "Hybrid + reranking lambda=0.7";
    byId("lambdaButtons").innerHTML = data.rerank.map((row, index) => {
      const lambda = row.Model.match(/lambda=([0-9.]+)/)?.[1] ?? "0.0";
      return `<button class="lambda-button ${index === data.rerank.length - 1 ? "active" : ""}" data-model="${esc(row.Model)}">${esc(lambda)}</button>`;
    }).join("");
  }

  function bindEvents() {
    ["metricSelect", "groupSelect", "sortSelect", "compareA", "compareB"].forEach((id) => {
      byId(id).addEventListener("change", renderModelExplorer);
    });
    byId("cvMetricSelect").addEventListener("change", renderCvSection);
    byId("feedbackMetricSelect").addEventListener("change", renderFeedbackLoop);
    document.querySelectorAll(".lambda-button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".lambda-button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        renderLambda(button.dataset.model);
      });
    });
    const navLinks = document.querySelectorAll("[data-section-link]");
    const sections = [...navLinks].map((link) => document.getElementById(link.dataset.sectionLink)).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        navLinks.forEach((link) => link.classList.toggle("active", link.dataset.sectionLink === entry.target.id));
      });
    }, { rootMargin: "-30% 0px -60% 0px", threshold: 0.01 });
    sections.forEach((section) => observer.observe(section));
  }

  function init() {
    renderHero();
    renderStoryline();
    renderMoviesDb();
    renderDatasetFoundation();
    populateControls();
    renderModelExplorer();
    renderCvSection();
    renderFeedbackLoop();
    renderLambda(data.rerank[data.rerank.length - 1].Model);
    renderProxyChart();
    renderEvidence();
    renderSources();
    bindEvents();
  }

  init();
}());
