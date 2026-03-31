const STORAGE_KEY = "cta-autoresearch-settings-v4";
const RUN_POLL_INTERVAL_MS = 2200;

const state = {
  controls: null,
  runs: [],
  activeRunId: null,
  selectedRunId: null,
  currentData: null,
  currentResultRunId: null,
  selectedPersonaIndex: 0,
  selectedCandidateId: null,
  filters: {
    message_angle: "all",
    proof_style: "all",
    offer: "all",
    cta: "all",
    personalization: "all",
  },
  pollTimer: null,
  liveApiReachable: true,
  asyncBackendReachable: true,
  lastLoadSource: "snapshot",
};

const fieldConfig = [
  { id: "setting-execution-mode", key: "execution_mode", type: "select" },
  { id: "setting-depth", key: "strategy_depth", type: "select" },
  { id: "setting-richness", key: "persona_richness", type: "select" },
  { id: "setting-model", key: "model_name", type: "select" },
  { id: "setting-population", key: "population", type: "range" },
  { id: "setting-ideation-agents", key: "ideation_agents", type: "range" },
  { id: "setting-validation-budget", key: "validation_budget", type: "range" },
  { id: "setting-discount-step", key: "discount_step", type: "range" },
  { id: "setting-discount-floor", key: "discount_floor", type: "range" },
  { id: "setting-discount-ceiling", key: "discount_ceiling", type: "range" },
  { id: "setting-grounding-limit", key: "grounding_limit", type: "range" },
  { id: "setting-treatment-limit", key: "treatment_limit", type: "range" },
  { id: "setting-friction-limit", key: "friction_limit", type: "range" },
  { id: "setting-ideas-per-agent", key: "idea_proposals_per_agent", type: "range" },
  { id: "setting-shortlist-multiplier", key: "persona_shortlist_multiplier", type: "range" },
  { id: "setting-segment-focus-limit", key: "segment_focus_limit", type: "range" },
  { id: "setting-api-batch-size", key: "api_batch_size", type: "range" },
  { id: "setting-reasoning-effort", key: "openai_reasoning_effort", type: "select" },
  { id: "setting-archetype-template-count", key: "archetype_template_count", type: "number" },
  { id: "setting-persona-blend-every", key: "persona_blend_every", type: "number" },
  { id: "setting-top-n", key: "top_n", type: "range" },
  { id: "setting-seed", key: "seed", type: "number" },
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function pct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function num(value, digits = 0) {
  if (value == null || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function titleize(value) {
  return String(value ?? "—")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatSeconds(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) {
    return "ETA calibrating";
  }
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) {
    return `ETA ${value}s`;
  }
  const minutes = Math.floor(value / 60);
  const remainder = value % 60;
  if (minutes < 60) {
    return `ETA ${minutes}m ${remainder}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `ETA ${hours}h ${minutes % 60}m`;
}

function formatTime(timestamp) {
  if (!timestamp) {
    return "—";
  }
  return new Date(timestamp * 1000 || timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function compressionLabel(meta) {
  const validated = Number(meta?.validated_strategy_count || 0);
  const universe = Number(meta?.search_space_size || 0);
  if (!validated || !universe) {
    return "—";
  }
  const ratio = validated / universe;
  return `${validated.toLocaleString()} / ${universe.toLocaleString()} (${pct(ratio)})`;
}

function apiJson(url, options = {}) {
  return fetch(url, {
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(options.headers || {}),
    },
    ...options,
  }).then(async (response) => {
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    return response.json();
  });
}

function buildSelectOptions(options, selectedValue) {
  return Object.entries(options || {})
    .map(([value, config]) => {
      const label = config?.label || titleize(value);
      const selected = String(value) === String(selectedValue) ? " selected" : "";
      return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(label)}</option>`;
    })
    .join("");
}

function detailCard(title, value, subtitle = "") {
  return `
    <article class="detail-card">
      <small>${escapeHtml(title)}</small>
      <strong>${escapeHtml(value)}</strong>
      ${subtitle ? `<p>${escapeHtml(subtitle)}</p>` : ""}
    </article>
  `;
}

function runStatusBadge(status) {
  const normalized = String(status || "unknown");
  return `<span class="score-pill">${escapeHtml(titleize(normalized))}</span>`;
}

function updateRangeValue(input) {
  const valueNode = document.getElementById(`${input.id}-value`);
  if (valueNode) {
    valueNode.textContent = input.value;
  }
}

function applyControlCatalog(controls) {
  if (!controls) {
    return;
  }
  state.controls = controls;
  const defaults = controls.defaults || {};
  const ranges = controls.advanced_control_ranges || {};

  const executionNode = document.getElementById("setting-execution-mode");
  const depthNode = document.getElementById("setting-depth");
  const richnessNode = document.getElementById("setting-richness");
  const modelNode = document.getElementById("setting-model");
  const reasoningNode = document.getElementById("setting-reasoning-effort");

  if (executionNode) {
    executionNode.innerHTML = buildSelectOptions(controls.execution_modes, defaults.execution_mode);
  }
  if (depthNode) {
    depthNode.innerHTML = buildSelectOptions(controls.depth_options, defaults.strategy_depth);
  }
  if (richnessNode) {
    richnessNode.innerHTML = buildSelectOptions(controls.persona_richness_options, defaults.persona_richness);
  }
  if (modelNode) {
    modelNode.innerHTML = buildSelectOptions(controls.model_options, defaults.model_name);
  }
  if (reasoningNode) {
    reasoningNode.innerHTML = buildSelectOptions(
      controls.reasoning_effort_options,
      defaults.openai_reasoning_effort,
    );
  }

  for (const config of fieldConfig) {
    const input = document.getElementById(config.id);
    if (!input) {
      continue;
    }
    const range = ranges[config.key];
    if (range) {
      if (range.min != null) {
        input.min = range.min;
      }
      if (range.max != null) {
        input.max = range.max;
      }
      if (range.step != null) {
        input.step = range.step;
      }
    }
    if (defaults[config.key] != null && !input.value) {
      input.value = defaults[config.key];
      updateRangeValue(input);
    }
  }
}

function loadPersistedSettings() {
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function persistSettings(settings) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

function applySettingsToForm(settings) {
  for (const config of fieldConfig) {
    const input = document.getElementById(config.id);
    if (!input) {
      continue;
    }
    const value = settings?.[config.key];
    if (value == null) {
      continue;
    }
    input.value = String(value);
    updateRangeValue(input);
  }
}

function gatherFormSettings() {
  const settings = {};
  for (const config of fieldConfig) {
    const input = document.getElementById(config.id);
    if (!input) {
      continue;
    }
    const rawValue = input.value;
    if (config.type === "range" || config.type === "number") {
      settings[config.key] = Number(rawValue);
    } else {
      settings[config.key] = rawValue;
    }
  }
  return settings;
}

function selectedRun() {
  return state.runs.find((run) => run.id === state.selectedRunId) || null;
}

function activeRun() {
  return state.runs.find((run) => run.id === state.activeRunId) || null;
}

function isPending(run) {
  return run && ["queued", "running"].includes(run.status);
}

function upsertRun(run) {
  const existingIndex = state.runs.findIndex((entry) => entry.id === run.id);
  if (existingIndex >= 0) {
    const merged = { ...state.runs[existingIndex], ...run };
    state.runs.splice(existingIndex, 1, merged);
  } else {
    state.runs.push(run);
  }
  state.runs.sort((left, right) => (right.created_at || 0) - (left.created_at || 0));
  if (run.status === "running" || run.status === "queued") {
    state.activeRunId = run.id;
  }
  if (run.id === state.activeRunId && !isPending(run) && state.activeRunId) {
    state.activeRunId = null;
  }
}

function makeSnapshotRun(data) {
  const meta = data?.meta || {};
  const settings = meta.research_settings || state.controls?.defaults || {};
  return {
    id: `snapshot-${Date.now()}`,
    status: "snapshot",
    progress: 1,
    stage: "snapshot",
    message: "Showing the last generated research snapshot.",
    created_at: Date.now() / 1000,
    started_at: Date.now() / 1000,
    finished_at: Date.now() / 1000,
    settings,
    result: data,
    result_meta: meta,
    activity_log: [
      {
        ts: Date.now() / 1000,
        stage: "snapshot",
        message: "Loaded dashboard snapshot.",
        progress: 1,
      },
    ],
    command_preview: "Static snapshot loaded from dashboard/data.json or cached /api/research output.",
  };
}

function setCurrentResult(data, runId = null) {
  if (!data) {
    return;
  }
  state.currentData = data;
  state.currentResultRunId = runId;
  state.selectedPersonaIndex = 0;
  state.selectedCandidateId = data.top_strategies?.[0]?.id || data.all_candidates?.[0]?.id || null;
}

function selectedResultData() {
  const run = selectedRun();
  if (run?.result) {
    return run.result;
  }
  return state.currentData;
}

function renderHero(data) {
  const meta = data?.meta || {};
  const metricsNode = document.getElementById("hero-metrics");
  metricsNode.innerHTML = [
    detailCard("Best Strategy Lift", pct(meta.top_lift), meta.top_strategy || "No strategy ranked yet."),
    detailCard("Top Composite Score", pct(meta.top_score), "Best strategy across retention, revenue, and trust."),
    detailCard("Candidates Validated", num(meta.validated_strategy_count), "Actual strategies scored in this run."),
    detailCard("Search-Space Compression", compressionLabel(meta), "How aggressively the lab compressed the possible universe."),
  ].join("");

  const statusNode = document.getElementById("run-status");
  const run = activeRun();
  const resultRun = selectedRun();
  const sourceLabel =
    state.lastLoadSource === "api"
      ? "Live API data"
      : state.lastLoadSource === "sync"
        ? "Fresh synchronous run"
        : "Generated snapshot";
  statusNode.innerHTML = [
    run
      ? `<span class="score-pill">Active: ${escapeHtml(titleize(run.stage || run.status))}</span>`
      : `<span class="score-pill">${escapeHtml(sourceLabel)}</span>`,
    `<span class="score-pill">Backend: ${escapeHtml(meta.model_backend || "unknown")}</span>`,
    resultRun ? `<span class="score-pill">Viewing: ${escapeHtml(resultRun.id)}</span>` : "",
  ].join("");
}

function renderRunSummary(data) {
  const meta = data?.meta || {};
  const settings = meta.research_settings || {};
  const controls = state.controls || data?.controls || {};
  const warningStack = document.getElementById("warning-stack");
  const summary = document.getElementById("run-summary");

  summary.innerHTML = [
    detailCard("Configured model", settings.model_name || "—", settings.execution_mode || "heuristic"),
    detailCard("Execution mode", settings.execution_mode || "—", meta.model_backend || "No backend marker"),
    detailCard("Backend scorer", meta.model_backend || "—", settings.uses_openai ? "OpenAI-backed path is enabled." : "Local heuristic scorer is active."),
    detailCard("Reasoning effort", settings.openai_reasoning_effort || "undefined", "Controls OpenAI ideation/evaluation depth."),
    detailCard("Segment focus limit", settings.segment_focus_limit ?? "undefined", "How many segments each ideation pass actively targets."),
    detailCard("API batch size", settings.api_batch_size ?? "undefined", "OpenAI judge batch size for API-based runs."),
    detailCard(
      "OpenAI API key",
      controls.openai_api_key_present ? "present" : "missing",
      controls.openai_api_key_present ? "Server environment is ready for live OpenAI calls." : "Hosted reruns will fall back or fail until the key is configured.",
    ),
  ].join("");

  const warnings = [...(meta.warnings || [])];
  if (!state.liveApiReachable) {
    if (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") {
      warnings.unshift("Live API was not reachable. The dashboard is showing the last generated snapshot, not a fresh rerun.");
    } else {
      warnings.unshift("This preview is serving the generated research snapshot. Live reruns require the lab server or a hosted API backend.");
    }
  }
  warningStack.innerHTML = warnings.length
    ? warnings.map((warning) => detailCard("Run warning", warning)).join("")
    : detailCard("Run health", "No warnings", "This result completed without reported backend warnings.");
}

function renderDimensions(data) {
  const dimensions = data?.dimensions || {};
  const node = document.getElementById("dimension-cards");
  node.innerHTML = Object.entries(dimensions)
    .map(([key, entries]) => {
      const items = Object.values(entries || {})
        .slice(0, 4)
        .map(
          (entry) =>
            `<li><strong>${escapeHtml(entry.label || "Unknown")}</strong><br /><small>${escapeHtml(entry.description || "No description")}</small></li>`,
        )
        .join("");
      return `
        <article class="dimension-card">
          <small>${escapeHtml(titleize(key))}</small>
          <h3>${escapeHtml(num(Object.keys(entries || {}).length))} options</h3>
          <ul>${items}</ul>
        </article>
      `;
    })
    .join("");
}

function renderTopPatterns(data) {
  const node = document.getElementById("top-patterns");
  const groups = data?.top_patterns || {};
  node.innerHTML = Object.entries(groups)
    .map(([key, entries]) => {
      const maxValue = Math.max(...Object.values(entries || { none: 0 }));
      const bars = Object.entries(entries || {})
        .map(([label, count]) => {
          const width = maxValue ? (Number(count) / maxValue) * 100 : 0;
          return `
            <article class="mini-bar">
              <small>${escapeHtml(titleize(key))}</small>
              <strong>${escapeHtml(label)}</strong>
              <span>${escapeHtml(String(count))} wins</span>
              <span class="bar-fill" style="width:${width}%"></span>
            </article>
          `;
        })
        .join("");
      return bars;
    })
    .join("");
}

function renderLeaderboard(data) {
  const body = document.getElementById("leaderboard-body");
  const strategies = data?.top_strategies || [];
  body.innerHTML = strategies
    .map(
      (strategy) => `
        <tr>
          <td>
            <strong>${escapeHtml(strategy.label)}</strong>
            <div><small>${escapeHtml(strategy.sample_message || "No sample message available.")}</small></div>
          </td>
          <td class="mono">${escapeHtml(pct(strategy.average_score))}</td>
          <td class="mono">${escapeHtml(pct(strategy.retention_score))}</td>
          <td class="mono">${escapeHtml(pct(strategy.revenue_score))}</td>
          <td class="mono">${escapeHtml(pct(strategy.trust_safety_score))}</td>
        </tr>
      `,
    )
    .join("");

  const nonDiscount = document.getElementById("best-non-discount");
  const best = data?.best_non_discount?.[0];
  nonDiscount.innerHTML = best
    ? detailCard("Best non-discount strategy", best.label, best.sample_message || "No sample message available.")
    : detailCard("Best non-discount strategy", "No non-discount winner", "Every leading candidate currently includes an offer.");
}

function renderSegments(data) {
  const node = document.getElementById("segment-grid");
  const segments = data?.segment_leaders || {};
  node.innerHTML = Object.entries(segments)
    .map(
      ([segment, leader]) => `
        <article class="segment-card">
          <small>${escapeHtml(titleize(segment))}</small>
          <h3>${escapeHtml(leader.label)}</h3>
          <p>${escapeHtml(leader.sample_message || "No sample message available.")}</p>
          <div class="stack">
            <span class="score-pill">Composite ${escapeHtml(pct(leader.average_score))}</span>
            <span class="score-pill">Lift ${escapeHtml(pct(leader.baseline_lift))}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderPersonas(data) {
  const personas = data?.personas || [];
  const listNode = document.getElementById("persona-list");
  const detailNode = document.getElementById("persona-detail");
  if (!personas.length) {
    listNode.innerHTML = `<div class="run-empty">No personas available for this run.</div>`;
    detailNode.innerHTML = "";
    return;
  }
  state.selectedPersonaIndex = Math.min(state.selectedPersonaIndex, personas.length - 1);
  const persona = personas[state.selectedPersonaIndex];

  listNode.innerHTML = personas
    .map(
      (entry, index) => `
        <button class="persona-chip ${index === state.selectedPersonaIndex ? "active" : ""}" data-persona-index="${index}">
          <strong>${escapeHtml(entry.name)}</strong>
          <div><small>${escapeHtml(titleize(entry.segment))}</small></div>
          <div><small>${escapeHtml(entry.study_context || "No context")}</small></div>
        </button>
      `,
    )
    .join("");

  const featureGrid = Object.entries(persona.features || {})
    .map(
      ([key, value]) => `
        <div class="profile-cell">
          <small>${escapeHtml(titleize(key))}</small>
          <strong>${escapeHtml(pct(value))}</strong>
        </div>
      `,
    )
    .join("");

  const rawProfile = Object.entries(persona.raw_profile || {})
    .slice(0, 12)
    .map(
      ([key, value]) => `
        <div class="profile-cell">
          <small>${escapeHtml(titleize(key))}</small>
          <strong>${escapeHtml(typeof value === "boolean" ? (value ? "true" : "false") : String(value))}</strong>
        </div>
      `,
    )
    .join("");

  detailNode.innerHTML = `
    <article class="detail-card">
      <small>${escapeHtml(persona.cohort)}</small>
      <h3>${escapeHtml(persona.name)}: ${escapeHtml(persona.profile_summary || "No summary")}</h3>
      <p>${escapeHtml(persona.dossier?.narrative || "No dossier narrative available.")}</p>
      <div class="stack">
        ${detailCard("Best strategy", persona.best_strategy?.label || "No strategy", persona.best_strategy?.sample_message || "")}
      </div>
    </article>
    <div class="feature-grid">${featureGrid}</div>
    <div class="profile-grid">${rawProfile}</div>
  `;

  listNode.querySelectorAll("[data-persona-index]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedPersonaIndex = Number(button.dataset.personaIndex);
      renderPersonas(selectedResultData());
    });
  });
}

function renderIdeaAgents(data) {
  const node = document.getElementById("idea-grid");
  const agents = data?.idea_agents || [];
  node.innerHTML = agents
    .map(
      (idea) => `
        <article class="detail-card">
          <small>${escapeHtml(idea.agent_role)}</small>
          <h3>${escapeHtml(idea.label)}</h3>
          <p>${escapeHtml(idea.thesis || "No thesis available.")}</p>
          <div class="stack">
            <span class="score-pill">Target ${escapeHtml(titleize(idea.target_segment || "unknown"))}</span>
            <span class="score-pill">Confidence ${escapeHtml(pct(idea.confidence))}</span>
          </div>
          <p>${escapeHtml(idea.sample_message || "No sample message available.")}</p>
        </article>
      `,
    )
    .join("");
}

function currentFilteredCandidates(data) {
  const candidates = data?.all_candidates || [];
  return candidates.filter((candidate) =>
    Object.entries(state.filters).every(([key, value]) => value === "all" || candidate[key] === value),
  );
}

function renderWorkbench(data) {
  const dimensions = data?.dimensions || {};
  const filterMap = [
    { id: "filter-message", key: "message_angle", source: dimensions.message_angles || {} },
    { id: "filter-proof", key: "proof_style", source: dimensions.proof_styles || {} },
    { id: "filter-offer", key: "offer", source: dimensions.offers || {} },
    { id: "filter-cta", key: "cta", source: dimensions.ctas || {} },
    { id: "filter-personalization", key: "personalization", source: dimensions.personalization || {} },
  ];

  filterMap.forEach(({ id, key, source }) => {
    const node = document.getElementById(id);
    if (!node.dataset.bound) {
      node.addEventListener("change", () => {
        state.filters[key] = node.value;
        renderWorkbench(selectedResultData());
      });
      node.dataset.bound = "true";
    }
    node.innerHTML = `<option value="all">All</option>${Object.entries(source)
      .map(([value, config]) => `<option value="${escapeHtml(value)}">${escapeHtml(config.label || titleize(value))}</option>`)
      .join("")}`;
    node.value = state.filters[key] || "all";
  });

  const filtered = currentFilteredCandidates(data);
  const resultCount = document.getElementById("result-count");
  resultCount.textContent = `${filtered.length} candidates match the current workbench filters.`;

  if (!filtered.some((candidate) => candidate.id === state.selectedCandidateId)) {
    state.selectedCandidateId = filtered[0]?.id || null;
  }
  const selectedCandidate = filtered.find((candidate) => candidate.id === state.selectedCandidateId) || filtered[0] || null;

  const body = document.getElementById("candidate-body");
  body.innerHTML = filtered
    .map(
      (candidate) => `
        <tr class="candidate-row ${candidate.id === state.selectedCandidateId ? "active" : ""}" data-candidate-id="${candidate.id}">
          <td>${escapeHtml(candidate.label)}</td>
          <td class="mono">${escapeHtml(pct(candidate.average_score))}</td>
          <td class="mono">${escapeHtml(pct(candidate.baseline_lift))}</td>
          <td class="mono">${escapeHtml(pct(candidate.trust_safety_score))}</td>
        </tr>
      `,
    )
    .join("");

  body.querySelectorAll("[data-candidate-id]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedCandidateId = row.dataset.candidateId;
      renderWorkbench(selectedResultData());
    });
  });

  const detail = document.getElementById("strategy-detail");
  detail.innerHTML = selectedCandidate
    ? `
      <article class="detail-card">
        <small>Selected candidate</small>
        <h3>${escapeHtml(selectedCandidate.label)}</h3>
        <p>${escapeHtml(selectedCandidate.sample_message || "No sample message available.")}</p>
      </article>
      <div class="feature-grid">
        ${detailCard("Composite", pct(selectedCandidate.average_score))}
        ${detailCard("Lift", pct(selectedCandidate.baseline_lift))}
        ${detailCard("Retention", pct(selectedCandidate.retention_score))}
        ${detailCard("Revenue", pct(selectedCandidate.revenue_score))}
        ${detailCard("Trust", pct(selectedCandidate.trust_safety_score))}
        ${detailCard("Offer", selectedCandidate.offer_label || "—")}
      </div>
    `
    : `<div class="run-empty">No candidates match the current filter combination.</div>`;
}

function renderResultSections(data) {
  renderHero(data);
  renderRunSummary(data);
  renderDimensions(data);
  renderTopPatterns(data);
  renderLeaderboard(data);
  renderSegments(data);
  renderPersonas(data);
  renderIdeaAgents(data);
  renderWorkbench(data);
}

function renderActiveRunCard(run) {
  const node = document.getElementById("active-run-card");
  if (!run) {
    node.innerHTML = `
      <article class="run-card run-empty">
        <strong>No active run</strong>
        <p>The dashboard is ready. Launch a deeper research run to see backend stages, ETA, and activity here.</p>
      </article>
    `;
    return;
  }
  const settings = run.settings || {};
  node.innerHTML = `
    <article class="run-card">
      <small class="run-kicker">Backend execution</small>
      <h3>Run ${escapeHtml(run.id)}</h3>
      <div class="stack">
        ${runStatusBadge(run.status)}
        <span class="score-pill">${escapeHtml(titleize(run.stage || "queued"))}</span>
      </div>
      <p>${escapeHtml(run.message || "Waiting for the backend to report progress.")}</p>
      <div class="feature-grid">
        ${detailCard("Execution mode", settings.execution_mode || "—")}
        ${detailCard("Model", settings.model_name || "—")}
        ${detailCard("Population", num(settings.population))}
        ${detailCard("Budget", num(settings.validation_budget))}
        ${detailCard("Depth", settings.strategy_depth || "—")}
        ${detailCard("Persona richness", settings.persona_richness || "—")}
      </div>
      <pre class="command-preview">${escapeHtml(run.command_preview || "Command preview unavailable.")}</pre>
    </article>
  `;
}

function renderRunTimeline(run) {
  const progressFill = document.getElementById("run-progress-fill");
  const progressMeta = document.getElementById("run-progress-meta");
  const timeline = document.getElementById("run-timeline");
  const progressValue = Math.max(0, Math.min(1, Number(run?.progress || 0)));
  progressFill.style.width = `${progressValue * 100}%`;

  if (!run) {
    progressMeta.textContent = "No research run is currently executing.";
    timeline.innerHTML = `<div class="run-empty">Activity will appear here once the backend accepts a run.</div>`;
    return;
  }

  progressMeta.innerHTML = `
    <span>${escapeHtml(pct(progressValue))} complete</span>
    <span>${escapeHtml(titleize(run.stage || run.status || "queued"))}</span>
    <span>${escapeHtml(formatSeconds(run.eta_seconds))}</span>
  `;

  const logEntries = [...(run.activity_log || [])].reverse();
  timeline.innerHTML = logEntries.length
    ? logEntries
        .map(
          (entry) => `
            <article class="log-line">
              <small>${escapeHtml(formatTime(entry.ts))} • ${escapeHtml(titleize(entry.stage || "event"))} • ${escapeHtml(pct(entry.progress || 0))}</small>
              <strong>${escapeHtml(entry.message || "No message")}</strong>
            </article>
          `,
        )
        .join("")
    : `<div class="run-empty">The backend has not emitted any activity yet.</div>`;
}

function runTabLabel(run) {
  if (run.status === "snapshot") {
    return "Snapshot";
  }
  return run.id;
}

function renderRunHistory() {
  const tabs = document.getElementById("run-history-tabs");
  const detail = document.getElementById("run-history-detail");
  const history = state.runs.filter((run) => run.status === "snapshot" || run.result || run.error || isPending(run));

  tabs.innerHTML = history.length
    ? history
        .map(
          (run) => `
            <button class="run-tab ${run.id === state.selectedRunId ? "active" : ""}" data-run-id="${run.id}">
              <small>${runStatusBadge(run.status)}</small>
              <strong>${escapeHtml(runTabLabel(run))}</strong>
              <div><small>${escapeHtml(formatTime(run.created_at))}</small></div>
            </button>
          `,
        )
        .join("")
    : `<div class="run-empty">No run history yet.</div>`;

  tabs.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      await selectRun(button.dataset.runId, { fetchDetail: true });
    });
  });

  const run = selectedRun();
  if (!run) {
    detail.innerHTML = `<div class="run-empty">Select a run to inspect its result and backend details.</div>`;
    return;
  }

  const meta = run.result_meta || run.result?.meta || {};
  const settings = run.settings || meta.research_settings || {};
  detail.innerHTML = `
    <article class="run-card">
      <small class="run-kicker">Selected run</small>
      <h3>${escapeHtml(runTabLabel(run))}</h3>
      <div class="stack">
        ${runStatusBadge(run.status)}
        <span class="score-pill">${escapeHtml(titleize(run.stage || run.status || "snapshot"))}</span>
      </div>
      <p>${escapeHtml(run.message || "No backend summary available.")}</p>
      <div class="feature-grid">
        ${detailCard("Elapsed", run.elapsed_seconds != null ? `${num(run.elapsed_seconds, 1)}s` : "—")}
        ${detailCard("Search-space compression", compressionLabel(meta))}
        ${detailCard("Backend scorer", meta.model_backend || "—")}
        ${detailCard("Model", settings.model_name || "—")}
        ${detailCard("Validation budget", num(settings.validation_budget))}
        ${detailCard("Top score", pct(meta.top_score))}
      </div>
      ${
        run.error
          ? `<article class="detail-card"><small>Run error</small><strong>${escapeHtml(run.error)}</strong></article>`
          : ""
      }
      <pre class="command-preview">${escapeHtml(run.command_preview || "Command preview unavailable.")}</pre>
    </article>
  `;
}

function renderRunCenter() {
  const run = activeRun() || selectedRun();
  renderActiveRunCard(activeRun());
  renderRunTimeline(run);
  renderRunHistory();
}

async function loadControls() {
  const dataControls = state.currentData?.controls;
  if (dataControls) {
    applyControlCatalog(dataControls);
  }
  try {
    const controls = await apiJson("/api/controls");
    state.liveApiReachable = true;
    applyControlCatalog(controls);
    return controls;
  } catch {
    state.liveApiReachable = false;
    return state.controls || dataControls || null;
  }
}

async function loadInitialData() {
  try {
    const payload = await apiJson("./data.json");
    state.lastLoadSource = "snapshot";
    return payload;
  } catch {
    const payload = await apiJson("/api/research");
    state.lastLoadSource = "api";
    return payload;
  }
}

async function runResearchSynchronously(settings) {
  const payload = await apiJson("/api/research", {
    method: "POST",
    body: JSON.stringify({ settings }),
  });
  state.lastLoadSource = "sync";
  return payload;
}

async function fetchRunList() {
  const payload = await apiJson("/api/research-runs");
  state.asyncBackendReachable = true;
  return payload.runs || [];
}

async function fetchRunDetail(runId) {
  return apiJson(`/api/research-runs/${encodeURIComponent(runId)}`);
}

async function createAsyncRun(settings) {
  const payload = await apiJson("/api/research-runs", {
    method: "POST",
    body: JSON.stringify({ settings }),
  });
  state.asyncBackendReachable = true;
  return payload;
}

function maybeStartPolling() {
  const pendingRuns = state.runs.filter(isPending);
  if (!pendingRuns.length) {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    return;
  }
  if (state.pollTimer) {
    return;
  }
  state.pollTimer = window.setInterval(async () => {
    try {
      const runs = await fetchRunList();
      runs.forEach(upsertRun);
      const pending = state.runs.filter(isPending);
      const detailTargets = new Set(
        [state.activeRunId, state.selectedRunId].filter((runId) =>
          pending.some((run) => run.id === runId) || state.runs.some((run) => run.id === runId && !run.result && run.error == null),
        ),
      );
      for (const runId of detailTargets) {
        const detail = await fetchRunDetail(runId);
        const before = state.runs.find((run) => run.id === runId);
        upsertRun(detail);
        if (detail.result && before && before.status !== "completed" && detail.status === "completed") {
          state.selectedRunId = detail.id;
          setCurrentResult(detail.result, detail.id);
          state.lastLoadSource = "api";
        }
      }
      renderRunCenter();
      if (state.currentData) {
        renderResultSections(selectedResultData());
      }
      maybeStartPolling();
    } catch {
      state.asyncBackendReachable = false;
      renderRunCenter();
    }
  }, RUN_POLL_INTERVAL_MS);
}

async function selectRun(runId, { fetchDetail = false } = {}) {
  let run = state.runs.find((entry) => entry.id === runId) || null;
  if (!run) {
    return;
  }
  if (fetchDetail && !run.result && !run.error && run.status !== "snapshot") {
    try {
      run = await fetchRunDetail(runId);
      upsertRun(run);
    } catch {
      /* ignore fetch failures and keep current summary */
    }
  }
  state.selectedRunId = runId;
  if (run.result) {
    setCurrentResult(run.result, run.id);
    state.lastLoadSource = "api";
    renderResultSections(run.result);
  }
  renderRunCenter();
}

function bindForm() {
  const form = document.getElementById("research-form");
  fieldConfig.forEach((config) => {
    const input = document.getElementById(config.id);
    if (!input) {
      return;
    }
    if (config.type === "range") {
      updateRangeValue(input);
      input.addEventListener("input", () => updateRangeValue(input));
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type='submit']");
    const settings = gatherFormSettings();
    persistSettings(settings);
    button.disabled = true;
    button.textContent = "Launching research...";
    try {
      try {
        const run = await createAsyncRun(settings);
        upsertRun(run);
        state.activeRunId = run.id;
        renderRunCenter();
        maybeStartPolling();
      } catch {
        state.asyncBackendReachable = false;
        const payload = await runResearchSynchronously(settings);
        setCurrentResult(payload);
        const snapshotRun = makeSnapshotRun(payload);
        upsertRun(snapshotRun);
        state.selectedRunId = snapshotRun.id;
        renderResultSections(payload);
        renderRunCenter();
      }
    } finally {
      button.disabled = false;
      button.textContent = "Run deeper research";
    }
  });
}

async function bootstrap() {
  const payload = await loadInitialData();
  setCurrentResult(payload);
  await loadControls();
  applyControlCatalog(payload.controls || state.controls);
  applySettingsToForm({
    ...(state.controls?.defaults || {}),
    ...(payload.meta?.research_settings || {}),
    ...loadPersistedSettings(),
  });

  const snapshotRun = makeSnapshotRun(payload);
  upsertRun(snapshotRun);
  state.selectedRunId = snapshotRun.id;
  renderResultSections(payload);
  renderRunCenter();
  bindForm();

  try {
    const runs = await fetchRunList();
    runs.forEach(upsertRun);
    const active = state.runs.find(isPending);
    if (active) {
      state.activeRunId = active.id;
      if (!state.selectedRunId) {
        state.selectedRunId = active.id;
      }
    }
    renderRunCenter();
    maybeStartPolling();
  } catch {
    state.asyncBackendReachable = false;
    renderRunCenter();
  }
}

bootstrap().catch((error) => {
  const runStatus = document.getElementById("run-status");
  runStatus.innerHTML = `<span class="score-pill">Dashboard failed to load</span>`;
  document.getElementById("warning-stack").innerHTML = detailCard("Load error", error.message || String(error));
});
