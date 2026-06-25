(function () {
  const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
  const pp = (value) => `${(Number(value) * 100).toFixed(1)} pp`;
  const dec = (value, digits = 4) => Number(value).toFixed(digits);
  const byId = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[ch]));

  const figuresBase = "../cultural_prominence_audit/outputs/final_notebook_figures/";

  const roadmapSteps = [
    {
      id: "framing",
      number: "01",
      title: "Frame the governance problem",
      short: "Catalogue diversity is not the same as ranked visibility.",
      question: "Why is this a governance problem and not only a recommendation benchmark?",
      why: "European works can be present in a catalogue while Top-K recommendation slots still concentrate attention on mainstream, English-language or US-linked items. The audit object is therefore ranked exposure.",
      technique: "State-of-the-art positioning: popularity bias, item-side exposure fairness, multimodal movie recommendation, feedback-loop dynamics and AVMSD/DSA governance framing.",
      outcome: "The research question becomes precise: do models under-expose European, non-English and long-tail films, and can mitigation improve prominence without hiding utility costs?",
      evidence: "State-of-art ledger and final research-question table.",
      figure: "08_visibility_funnel_baselines.png",
      caption: "The visibility funnel shows why catalogue share, user history, relevant test share and Top-K exposure are separate layers."
    },
    {
      id: "data",
      number: "02",
      title: "Build the audit dataset",
      short: "Join MovieLens/M3L interactions with Wikidata cultural labels.",
      question: "Can we create a defensible movie-level table for country and language analysis?",
      why: "A cultural audit is only as strong as its identifiers and missingness reporting. We therefore preserve unknown labels instead of silently dropping them.",
      technique: "MovieLens movieId bridge, M3L interaction and MPNet/CLIP feature coverage, cached Wikidata labels for country of origin, original language, production company and release metadata.",
      outcome: "The notebook documents metadata coverage, feature coverage, join funnel and proxy risk before any model result is interpreted.",
      evidence: "Join funnel, metadata coverage and labelled movie preview tables.",
      figure: "03_join_funnel_final.png",
      caption: "The join funnel makes data loss visible before the model stage."
    },
    {
      id: "labels",
      number: "03",
      title: "Define cultural proxies carefully",
      short: "Separate country, language, co-production and company involvement.",
      question: "What exactly do we mean by European, non-English, Spain-origin and Spanish-language?",
      why: "Country and language are not natural facts. Co-productions, multilingual films and US-company involvement can change the interpretation.",
      technique: "Binary labels for headline groups, fractional credit for country/language distributions, wider-Europe and EU27 sensitivity, explicit unknown-country and unknown-language buckets.",
      outcome: "The notebook can ask finer questions without pretending the proxy labels are perfect cultural identity labels.",
      evidence: "Catalogue country/language distributions, Spain-origin vs Spanish-language counts and proxy-risk ledgers.",
      figure: "06_spain_origin_vs_spanish_language.png",
      caption: "Spain-origin and Spanish-language are related but not the same audit group."
    },
    {
      id: "models",
      number: "04",
      title: "Train comparable recommenders",
      short: "Use transparent baselines plus multimodal content models.",
      question: "Do different recommender families produce different cultural prominence patterns?",
      why: "A single algorithm cannot tell us whether the effect is structural. We need baselines and multimodal alternatives to see whether visibility changes with model logic.",
      technique: "Popularity baseline, ItemKNN, TruncatedSVD, MPNet-content, CLIP-image-content and a simple Hybrid model. Utility is measured with NDCG@20, Recall@20 and MAP@20.",
      outcome: "SVD has the best utility in the bounded run, but CLIP-image-content has the highest European exposure. Utility and cultural visibility are not the same outcome.",
      evidence: "Model utility metrics and aggregate visibility metrics.",
      figure: "11_accuracy_vs_european_exposure.png",
      caption: "The scatter plot shows that accuracy and European exposure occupy different axes."
    },
    {
      id: "geo",
      number: "05",
      title: "Audit country visibility",
      short: "Find where Europe is visible and where it disappears.",
      question: "Which European countries are most and least visible in Top-K rankings?",
      why: "A Europe-wide label can hide intra-European concentration. The governance question becomes stronger when we can point to country-level gaps.",
      technique: "Country-level Exposure@20 and PACPG@20. The target is max(user-history share, relevant-test share), so underexposure is judged relative to observed user interest.",
      outcome: "United Kingdom is the most visible support-passing country. France has the lowest mean PACPG@20; Poland, Netherlands, Switzerland, Sweden and Austria are near-invisible despite a target signal.",
      evidence: "Country scorecard, target-vs-exposure plot and country-border choropleth.",
      figure: "12_country_geo_scorecard.png",
      caption: "The scorecard turns country-level PACPG into an easy presentation result."
    },
    {
      id: "feedback",
      number: "06",
      title: "Stress-test feedback-loop drift",
      short: "Check whether repeated recommendations change representation.",
      question: "Can country/language gaps compound through repeated recommendation?",
      why: "Schedl/Lesota-style feedback-loop work shows that representation can drift when recommendations are repeatedly accepted and fed back into profiles.",
      technique: "Schedl-inspired lightweight offline stress test: repeated Top-K recommendations, rank-biased accepted item, profile/seen-mask update, but no full per-iteration retraining inside the bounded notebook.",
      outcome: "Largest reported loss is Popularity / Spanish-language (-1.4 pp); largest reported gain is Popularity / Europe wide (+3.7 pp). We interpret this as dynamic stress-test evidence, not causal platform evidence.",
      evidence: "Feedback-loop methodology comparison and exposure drift summary.",
      figure: "18_feedback_loop_exposure.png",
      caption: "The feedback-loop chart shows exposure drift across repeated recommendation steps."
    },
    {
      id: "mitigation",
      number: "07",
      title: "Report mitigation as a trade-off",
      short: "Make re-ranking pressure visible instead of hiding it.",
      question: "Can governance-aware re-ranking improve visibility without pretending there is no utility cost?",
      why: "A platform-facing audit should not silently moralise the ranking. It should quantify how much utility is spent for prominence improvement.",
      technique: "Transparent post-processing: final_score = relevance_score + lambda * prominence_gain. Lambda sweep reports NDCG, Recall, exposure and PACPG together.",
      outcome: "Strict 80% NDCG retention selects lambda=0.0 in the current run; maximum visibility occurs at lambda=0.4 with 100.0% Europe exposure but much lower NDCG@20.",
      evidence: "Re-ranking frontier and model comparison tables.",
      figure: "19_reranking_frontier.png",
      caption: "The frontier is the governance output: visibility improvement always comes with an explicit utility reading."
    },
    {
      id: "dna",
      number: "08",
      title: "Extend the research with Visibility DNA",
      short: "Ask whether rankings preserve local Europe or mainly global-compatible Europe.",
      question: "Does the recommender show Europe as local culture, or only the globally compatible version of Europe?",
      why: "A Europe-wide label can hide whether rankings surface non-English, low-US-involvement, local European films or mostly English-language, industrially global and highly compatible European productions.",
      technique: "Transparent Global Compatibility Score and Local Europe Score, built from language, origin, US involvement, blockbuster status, interaction signal and co-production structure. A limited deep-Wikidata cache adds director citizenship, filming-country and award-count context.",
      outcome: "Platform-compatible Europe has mean Exposure@20 of 18.1%; Local Europe has mean Exposure@20 of 0.6%; the stricter local non-English/no-US subset reaches only 0.8%.",
      evidence: "Visibility DNA group metrics, score distribution, group exposure and PACPG figures.",
      figure: "21_visibility_dna_group_exposure.png",
      caption: "The extension shows that local Europe is much less visible than globally compatible Europe in the current run."
    }
  ];

  const countryRisks = [
    {
      name: "France",
      pacpg: "-1.7 pp",
      exposure: "3.0%",
      target: "4.7%",
      type: "underexposed relative to target",
      note: "Strongest support-passing underexposure signal by mean PACPG@20."
    },
    {
      name: "Spain",
      pacpg: "-0.5 pp",
      exposure: "0.1%",
      target: "0.6%",
      type: "near target or low-support signal",
      note: "Visible mainly through MPNet-content; still weak as Spain-origin exposure."
    },
    {
      name: "Germany",
      pacpg: "-0.4 pp",
      exposure: "2.8%",
      target: "3.2%",
      type: "near target or low-support signal",
      note: "Not invisible, but below its preference-adjusted target on average."
    },
    {
      name: "Poland",
      pacpg: "-0.3 pp",
      exposure: "0.0%",
      target: "0.3%",
      type: "near-invisible despite target signal",
      note: "Small absolute target, but the ranking signal is nearly absent."
    },
    {
      name: "United Kingdom",
      pacpg: "+0.4 pp",
      exposure: "13.8%",
      target: "13.4%",
      type: "most visible",
      note: "Highest mean Exposure@20 among support-passing European countries."
    }
  ];

  const modelRows = [
    { model: "Popularity", ndcg: 0.118636, europe: 0.097407, nonEnglish: 0.000087, pacpg: -0.103187 },
    { model: "ItemKNN", ndcg: 0.162323, europe: 0.148670, nonEnglish: 0.002903, pacpg: -0.051924 },
    { model: "SVD", ndcg: 0.181420, europe: 0.187734, nonEnglish: 0.021240, pacpg: -0.012860 },
    { model: "MPNet-content", ndcg: 0.019410, europe: 0.181217, nonEnglish: 0.087510, pacpg: -0.019377 },
    { model: "CLIP-image-content", ndcg: 0.014402, europe: 0.327980, nonEnglish: 0.009676, pacpg: 0.127386 },
    { model: "Hybrid", ndcg: 0.166821, europe: 0.182289, nonEnglish: 0.013167, pacpg: -0.018305 }
  ];

  const metrics = [
    {
      key: "ndcg",
      label: "NDCG@20",
      title: "Utility: SVD is the strongest ranking baseline",
      reading: "SVD has the best NDCG@20 in the bounded notebook run. This is why utility is reported separately from cultural prominence.",
      format: dec,
      notes: [
        ["Why this metric", "NDCG@20 rewards relevant test items near the top of the ranking."],
        ["Presentation line", "The best utility model is not automatically the best cultural-visibility model."]
      ]
    },
    {
      key: "europe",
      label: "Europe Exposure@20",
      title: "European exposure: CLIP shifts visibility the most",
      reading: "CLIP-image-content reaches the highest European Exposure@20. That makes multimodal comparison useful, even though its utility is weaker.",
      format: pct,
      notes: [
        ["Why this metric", "Rank-discounted group exposure measures what appears in visible Top-K positions."],
        ["Presentation line", "Model family changes cultural prominence, not only accuracy."]
      ]
    },
    {
      key: "nonEnglish",
      label: "Non-English Exposure@20",
      title: "Language exposure: MPNet is strongest for non-English visibility",
      reading: "MPNet-content has the highest non-English exposure. Text features can surface language-space signals that collaborative baselines barely expose.",
      format: pct,
      notes: [
        ["Why this metric", "It checks language visibility separately from production country."],
        ["Presentation line", "Country and language must not be collapsed into one label."]
      ]
    },
    {
      key: "pacpg",
      label: "Europe PACPG@20",
      title: "PACPG: Popularity has the largest Europe gap",
      reading: "Popularity has the worst Europe PACPG@20. PACPG is the central audit metric because it compares exposure to a user-adjusted target.",
      format: pp,
      notes: [
        ["Why this metric", "PACPG = Exposure@K minus max(history share, relevant-test share)."],
        ["Presentation line", "The gap is not just catalogue share; it is visibility relative to observed user interest."]
      ]
    }
  ];

  const answers = [
    ["Does the algorithm hide Europe?", "Model-dependent: Popularity has the worst Europe PACPG@20 (-10.3 pp), while CLIP-image-content has the highest European exposure (32.8%).", "Caveat: offline sampled audit with proxy metadata."],
    ["Is catalogue diversity enough?", "No. Catalogue share, interactions, user history, relevant test share and ranked exposure are different layers.", "Caveat: metadata coverage still matters."],
    ["Which Europe gets recommended?", "Visibility concentrates unevenly: UK/Ireland is strongest; France is the weakest country signal by PACPG.", "Caveat: low-support regions are not overclaimed."],
    ["Local Europe or globally compatible Europe?", "Platform-compatible Europe reaches 18.1% mean Exposure@20; Local Europe reaches 0.6%; the stricter local non-English/no-US subset reaches 0.8%.", "Caveat: transparent DNA proxy layer; TMDb/LUMIERE are documented next layers, not faked."],
    ["Do models differ?", "Yes. SVD leads utility, CLIP leads European exposure, MPNet leads non-English exposure.", "Caveat: bounded candidate and user sample."],
    ["Does feedback-loop drift matter?", "The Schedl-inspired stress test reports dynamic drift; Spanish-language loses visibility under Popularity in the current run.", "Caveat: stress test, not causal platform evidence."],
    ["Can re-ranking mitigate it?", "Yes, but only as a trade-off frontier: stronger lambda improves visibility while reducing utility.", "Caveat: re-ranking does not solve fairness by itself."]
  ];

  const methods = [
    {
      tag: "Data",
      title: "MovieLens + M3L + Wikidata",
      body: "MovieLens gives interaction behaviour, M3L gives multimodal features, and Wikidata supplies country/language proxy labels. This combination makes cultural visibility auditable."
    },
    {
      tag: "Models",
      title: "Popularity, ItemKNN, SVD, MPNet, CLIP, Hybrid",
      body: "The model set is intentionally mixed: transparent baselines, collaborative filtering, text/image content models and a simple hybrid for comparison."
    },
    {
      tag: "Metric",
      title: "PACPG",
      body: "Preference-Adjusted Cultural Prominence Gap asks whether ranked exposure is lower than a transparent target adjusted for observed user interest."
    },
    {
      tag: "Robustness",
      title: "User-fold stability check",
      body: "The notebook calls this a bounded robustness check, not full retraining cross-validation, which keeps the method defensible."
    },
    {
      tag: "Dynamics",
      title: "Schedl-inspired feedback loop",
      body: "Repeated recommendations and profile updates stress-test whether visibility can drift over time, with explicit limits documented."
    },
    {
      tag: "Mitigation",
      title: "Lambda re-ranking",
      body: "The re-ranker makes intervention strength explicit and reports utility retention and visibility improvement together."
    },
    {
      tag: "Extension",
      title: "European Film Visibility DNA",
      body: "The extension asks whether rankings show local, non-English, low-US-involvement Europe or mainly globally compatible European films."
    }
  ];

  const dnaCards = [
    {
      value: "18.1%",
      label: "Platform-compatible Europe exposure",
      text: "English-language, US-involved, blockbuster or highly interaction-compatible European films receive the strongest DNA-group exposure."
    },
    {
      value: "0.6%",
      label: "Local Europe exposure",
      text: "Local Europe is much less visible: European, non-English, no-US-involvement and non-blockbuster signals rarely survive the ranking layer."
    },
    {
      value: "475",
      label: "Deep-Wikidata priority rows",
      text: "The optional DNA cache adds real director citizenship, filming-location and award-count context for a priority subset; no TMDb or LUMIERE data is faked."
    }
  ];

  function renderRoadmap() {
    const list = byId("stepList");
    list.innerHTML = roadmapSteps.map((step, index) => `
      <button class="step-button ${index === 0 ? "active" : ""}" type="button" data-step="${esc(step.id)}">
        <span class="step-index">${esc(step.number)}</span>
        <strong>${esc(step.title)}</strong>
        <small>${esc(step.short)}</small>
      </button>
    `).join("");

    list.addEventListener("click", (event) => {
      const button = event.target.closest("[data-step]");
      if (!button) return;
      list.querySelectorAll(".step-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderStep(button.dataset.step);
    });

    renderStep(roadmapSteps[0].id);
  }

  function renderStep(id) {
    const step = roadmapSteps.find((item) => item.id === id) || roadmapSteps[0];
    byId("stepDetail").innerHTML = `
      <div class="step-detail-grid">
        <div class="step-detail-copy">
          <span class="step-index">${esc(step.number)}</span>
          <h3>${esc(step.title)}</h3>
          <p>${esc(step.short)}</p>
          <div class="step-meta">
            ${metaBlock("Question answered", step.question)}
            ${metaBlock("Why we did it", step.why)}
            ${metaBlock("Technique used", step.technique)}
            ${metaBlock("Outcome", step.outcome)}
            ${metaBlock("Evidence artifact", step.evidence)}
          </div>
        </div>
        <figure class="visual-frame">
          <img src="${figuresBase}${esc(step.figure)}" alt="${esc(step.title)} evidence figure" />
          <figcaption>${esc(step.caption)}</figcaption>
        </figure>
      </div>
    `;
  }

  function metaBlock(label, text) {
    return `
      <div class="meta-block">
        <strong>${esc(label)}</strong>
        <p>${esc(text)}</p>
      </div>
    `;
  }

  function renderCountryRisk() {
    byId("riskList").innerHTML = countryRisks.map((item) => `
      <article class="risk-item ${item.pacpg.startsWith("+") ? "positive" : ""}">
        <header>
          <strong>${esc(item.name)}</strong>
          <span>${esc(item.pacpg)}</span>
        </header>
        <p>${esc(item.note)}</p>
        <p><strong>Exposure:</strong> ${esc(item.exposure)} &nbsp; <strong>Target:</strong> ${esc(item.target)}</p>
        <p>${esc(item.type)}</p>
      </article>
    `).join("");
  }

  function renderMetricButtons() {
    const wrap = byId("metricButtons");
    wrap.innerHTML = metrics.map((metric, index) => `
      <button type="button" class="metric-button ${index === 0 ? "active" : ""}" data-metric="${esc(metric.key)}">
        ${esc(metric.label)}
      </button>
    `).join("");

    wrap.addEventListener("click", (event) => {
      const button = event.target.closest("[data-metric]");
      if (!button) return;
      wrap.querySelectorAll(".metric-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderModelChart(button.dataset.metric);
    });

    renderModelChart(metrics[0].key);
  }

  function renderModelChart(metricKey) {
    const metric = metrics.find((item) => item.key === metricKey) || metrics[0];
    const values = modelRows.map((row) => row[metric.key]);
    const min = Math.min(0, ...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const sorted = [...modelRows].sort((a, b) => b[metric.key] - a[metric.key]);

    byId("modelChart").innerHTML = sorted.map((row) => {
      const value = row[metric.key];
      const width = Math.max(2, Math.abs(value) / Math.max(Math.abs(max), Math.abs(min), 0.001) * 100);
      return `
        <div class="bar-row">
          <span>${esc(row.model)}</span>
          <div class="bar-track" title="${esc(metric.label)}">
            <div class="bar-fill ${value < 0 ? "negative" : ""}" style="width:${width}%"></div>
          </div>
          <div class="bar-value">${esc(metric.format(value))}</div>
        </div>
      `;
    }).join("");

    byId("metricTitle").textContent = metric.title;
    byId("metricReading").textContent = metric.reading;
    byId("metricNotes").innerHTML = metric.notes.map(([dt, dd]) => `
      <div>
        <dt>${esc(dt)}</dt>
        <dd>${esc(dd)}</dd>
      </div>
    `).join("");
  }

  function renderAnswers() {
    byId("answerGrid").innerHTML = answers.map(([question, answer, caveat]) => `
      <article class="answer-card">
        <strong>${esc(question)}</strong>
        <p>${esc(answer)}</p>
        <em>${esc(caveat)}</em>
      </article>
    `).join("");
  }

  function renderMethods() {
    byId("methodGrid").innerHTML = methods.map((method) => `
      <article class="method-card">
        <span>${esc(method.tag)}</span>
        <h3>${esc(method.title)}</h3>
        <p>${esc(method.body)}</p>
      </article>
    `).join("");
  }

  function renderDNA() {
    const host = byId("dnaGrid");
    if (!host) return;
    host.innerHTML = dnaCards.map((card) => `
      <article class="dna-card">
        <strong>${esc(card.value)}</strong>
        <span>${esc(card.label)}</span>
        <p>${esc(card.text)}</p>
      </article>
    `).join("");
  }

  renderRoadmap();
  renderCountryRisk();
  renderMetricButtons();
  renderAnswers();
  renderMethods();
  renderDNA();
}());
