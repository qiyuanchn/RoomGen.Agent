/* RoomGen.Agent PointNav replay.
 *
 * Plain browser JavaScript, no modules and no network requests: the generated
 * replay-data.js defines window.NAV_REPLAY_DATA and this file renders it with
 * Canvas 2D.  A top-down map shows room polygons, the reference path, the
 * executed trajectory, endpoints, collisions and failure evidence.
 */
(() => {
  'use strict';

  const DATA = window.NAV_REPLAY_DATA;
  const CATEGORY_LABELS = (DATA && DATA.categories) || {
    success: 'Success',
    stuck_door: 'Stuck at door',
    stuck: 'Stuck / blocked',
    through_wall: 'Through wall',
    cross_void: 'Crossed void / hole',
    floor_jump: 'Wrong floor jump',
    collision: 'Collision',
    timeout: 'Timeout',
    other: 'Other failure',
  };
  const CATEGORY_COLORS = {
    success: '#5dd39e',
    stuck_door: '#c792ea',
    stuck: '#c792ea',
    through_wall: '#ff6b9d',
    cross_void: '#ffb454',
    floor_jump: '#57c7ff',
    collision: '#ff7b72',
    timeout: '#d0b36a',
    other: '#9aa7b6',
  };
  const EVENT_COLORS = {
    collision: '#ff7b72',
    stuck_door: '#c792ea',
    stuck: '#c792ea',
    through_wall: '#ff6b9d',
    cross_void: '#ffb454',
    floor_jump: '#57c7ff',
    start: '#5dd39e',
    goal: '#ff7b72',
  };
  const FLOOR_COLORS = [
    '#6fb2ff', '#5dd39e', '#c792ea', '#ffb454',
    '#57c7ff', '#ff6b9d', '#9ad0a8', '#d0b36a',
  ];
  const CATEGORY_ORDER = [
    'success', 'stuck_door', 'stuck', 'through_wall', 'cross_void',
    'floor_jump', 'collision', 'timeout', 'other',
  ];

  let episodes = [];
  let scenes = {};
  let els = null;
  let rafHandle = 0;
  let ctx = null;
  let dpr = 1;

  const state = {
    scene: 'all',
    result: 'all',
    category: 'all',
    epKey: null,
    floor: 'all',
    step: 0,
    playing: false,
    speed: 1,
    selectedEvent: -1,
    width: 0,
    height: 0,
    zoom: 1,
    panX: 0,
    panY: 0,
    lastTs: 0,
    acc: 0,
    hitMarkers: [],
    drag: null,
  };

  // ----------------------------------------------------------------------- //
  // setup
  // ----------------------------------------------------------------------- //
  function fatal(message) {
    const box = document.createElement('div');
    box.className = 'canvas-empty';
    box.style.position = 'fixed';
    box.style.inset = '0';
    box.textContent = message;
    document.body.replaceChildren(box);
  }

  function init() {
    if (!DATA || !Array.isArray(DATA.episodes)) {
      fatal('No NAV_REPLAY_DATA found. Serve the generated site directory.');
      return;
    }
    episodes = DATA.episodes.slice();
    scenes = DATA.scenes || {};
    if (DATA.title) document.title = `${DATA.title} · PointNav replay`;
    els = {
      runMeta: document.getElementById('run-meta'),
      sceneFilter: document.getElementById('scene-filter'),
      resultFilter: document.getElementById('result-filter'),
      categoryFilter: document.getElementById('category-filter'),
      resetFilters: document.getElementById('reset-filters'),
      episodeCount: document.getElementById('episode-count'),
      episodeList: document.getElementById('episode-list'),
      globalMetrics: document.getElementById('global-metrics'),
      episodeTitle: document.getElementById('episode-title'),
      episodeSubtitle: document.getElementById('episode-subtitle'),
      resultBadge: document.getElementById('result-badge'),
      floorSelect: document.getElementById('floor-select'),
      zoomReset: document.getElementById('zoom-reset'),
      floorLegend: document.getElementById('floor-legend'),
      canvasWrap: document.getElementById('canvas-wrap'),
      canvas: document.getElementById('map-canvas'),
      canvasEmpty: document.getElementById('canvas-empty'),
      tooltip: document.getElementById('map-tooltip'),
      playButton: document.getElementById('play-button'),
      stepBack: document.getElementById('step-back'),
      stepForward: document.getElementById('step-forward'),
      restartButton: document.getElementById('restart-button'),
      endButton: document.getElementById('end-button'),
      stepSlider: document.getElementById('step-slider'),
      stepLabel: document.getElementById('step-label'),
      actionLabel: document.getElementById('action-label'),
      speedSelect: document.getElementById('speed-select'),
      episodeMetrics: document.getElementById('episode-metrics'),
      failureBox: document.getElementById('failure-box'),
      eventCount: document.getElementById('event-count'),
      eventList: document.getElementById('event-list'),
      warningsPanel: document.getElementById('warnings-panel'),
      warnings: document.getElementById('warnings'),
      footerInfo: document.getElementById('footer-info'),
      footerHint: document.getElementById('footer-hint'),
    };
    ctx = els.canvas.getContext('2d');
    populateFilters();
    bindEvents();
    renderRunMeta();
    renderGlobalMetrics();
    renderFooter();
    applyFilters(true);
    if (typeof ResizeObserver === 'function') {
      new ResizeObserver(() => {
        if (resizeCanvas()) draw();
      }).observe(els.canvasWrap);
    } else {
      window.addEventListener('resize', () => {
        if (resizeCanvas()) draw();
      });
    }
  }

  function bindEvents() {
    els.sceneFilter.addEventListener('change', () => {
      state.scene = els.sceneFilter.value;
      applyFilters();
    });
    els.resultFilter.addEventListener('change', () => {
      state.result = els.resultFilter.value;
      applyFilters();
    });
    els.categoryFilter.addEventListener('change', () => {
      state.category = els.categoryFilter.value;
      applyFilters();
    });
    els.resetFilters.addEventListener('click', () => {
      state.scene = 'all';
      state.result = 'all';
      state.category = 'all';
      els.sceneFilter.value = 'all';
      els.resultFilter.value = 'all';
      els.categoryFilter.value = 'all';
      applyFilters();
    });
    els.floorSelect.addEventListener('change', () => {
      state.floor = els.floorSelect.value;
      state.selectedEvent = -1;
      resetView();
      renderEvents(currentEpisode());
      draw();
    });
    els.zoomReset.addEventListener('click', () => {
      resetView();
      draw();
    });
    els.playButton.addEventListener('click', togglePlayback);
    els.stepBack.addEventListener('click', () => setStep(state.step - 1));
    els.stepForward.addEventListener('click', () => setStep(state.step + 1));
    els.restartButton.addEventListener('click', () => setStep(0));
    els.endButton.addEventListener('click', () => {
      const ep = currentEpisode();
      if (ep && ep.has_trajectory) setStep(ep.trajectory.length - 1);
    });
    els.stepSlider.addEventListener('input', () => {
      setStep(parseInt(els.stepSlider.value, 10) || 0);
    });
    els.speedSelect.addEventListener('change', () => {
      state.speed = parseFloat(els.speedSelect.value) || 1;
    });

    els.canvas.addEventListener('wheel', onWheel, { passive: false });
    els.canvas.addEventListener('pointerdown', onPointerDown);
    els.canvas.addEventListener('pointermove', onPointerMove);
    els.canvas.addEventListener('pointerup', onPointerUp);
    els.canvas.addEventListener('pointercancel', onPointerUp);
    els.canvas.addEventListener('pointerleave', () => {
      state.drag = null;
      els.canvasWrap.classList.remove('is-dragging');
      hideTooltip();
    });
    els.canvas.addEventListener('dblclick', () => {
      resetView();
      draw();
    });

    window.addEventListener('keydown', (event) => {
      const tag = (event.target && event.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'BUTTON') {
        return;
      }
      if (event.code === 'Space') {
        event.preventDefault();
        togglePlayback();
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        setStep(state.step - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        setStep(state.step + 1);
      } else if (event.key === 'Home') {
        event.preventDefault();
        setStep(0);
      } else if (event.key === 'End') {
        const ep = currentEpisode();
        if (ep && ep.has_trajectory) setStep(ep.trajectory.length - 1);
      } else if (event.key === 'r' || event.key === 'R') {
        resetView();
        draw();
      }
    });
  }

  // ----------------------------------------------------------------------- //
  // filters and episode list
  // ----------------------------------------------------------------------- //
  function populateFilters() {
    const sceneCounts = new Map();
    for (const episode of episodes) {
      sceneCounts.set(episode.scene_id, (sceneCounts.get(episode.scene_id) || 0) + 1);
    }
    els.sceneFilter.replaceChildren(option('all', `All scenes (${episodes.length})`));
    for (const sceneId of [...sceneCounts.keys()].sort()) {
      els.sceneFilter.appendChild(
        option(sceneId, `${sceneId} (${sceneCounts.get(sceneId)})`)
      );
    }

    const present = [...new Set(episodes.map((episode) => episode.failure_category))]
      .filter(Boolean);
    present.sort((a, b) => {
      const ai = CATEGORY_ORDER.indexOf(a);
      const bi = CATEGORY_ORDER.indexOf(b);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || a.localeCompare(b);
    });
    els.categoryFilter.replaceChildren(option('all', 'All types'));
    for (const category of present) {
      els.categoryFilter.appendChild(
        option(category, CATEGORY_LABELS[category] || category)
      );
    }
  }

  function option(value, label) {
    const node = document.createElement('option');
    node.value = value;
    node.textContent = label;
    return node;
  }

  function filteredEpisodes() {
    return episodes.filter((episode) => {
      if (state.scene !== 'all' && episode.scene_id !== state.scene) return false;
      if (state.result === 'success' && episode.success !== true) return false;
      if (state.result === 'failure' && episode.success !== false) return false;
      if (state.category !== 'all' && episode.failure_category !== state.category) {
        return false;
      }
      return true;
    });
  }

  function applyFilters(selectFirst) {
    const filtered = filteredEpisodes();
    els.episodeCount.textContent = `${filtered.length} / ${episodes.length} episodes`;
    renderEpisodeList(filtered);
    if (!filtered.length) {
      selectEpisode(null);
      return;
    }
    const current = filtered.find((episode) => episode.key === state.epKey);
    if (!current) {
      selectEpisode(filtered[0].key);
    } else if (selectFirst) {
      selectEpisode(current.key);
    } else {
      updateListSelection();
    }
  }

  function renderEpisodeList(filtered) {
    els.episodeList.replaceChildren();
    if (!filtered.length) {
      const note = document.createElement('li');
      note.className = 'empty-note';
      note.textContent = 'No episodes match these filters.';
      els.episodeList.appendChild(note);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const episode of filtered) {
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'episode-item';
      button.dataset.key = episode.key;
      if (episode.key === state.epKey) button.classList.add('is-selected');
      button.title = failureText(episode) || '';

      const result = document.createElement('span');
      result.className = `ep-result ${
        episode.success === true ? 'success'
          : episode.success === false ? 'failure' : 'unknown'
      }`;
      result.textContent = episode.success === true ? '✓'
        : episode.success === false ? '✕' : '?';

      const main = document.createElement('span');
      main.className = 'ep-main';
      const id = document.createElement('span');
      id.className = 'ep-id';
      id.textContent = `#${episode.episode_id}`;
      const scene = document.createElement('span');
      scene.className = 'ep-scene';
      scene.textContent = episode.scene_id;
      main.append(id, scene);

      const meta = document.createElement('span');
      meta.className = 'ep-meta';
      const category = document.createElement('span');
      category.className = `ep-cat cat-${episode.failure_category || 'other'}`;
      category.textContent = CATEGORY_LABELS[episode.failure_category]
        || episode.failure_category || 'unknown';
      const spl = document.createElement('span');
      spl.textContent = `SPL ${format(episode.spl, 2)}`;
      meta.append(category, spl);

      button.append(result, main, meta);
      button.addEventListener('click', () => selectEpisode(episode.key));
      item.appendChild(button);
      fragment.appendChild(item);
    }
    els.episodeList.appendChild(fragment);
  }

  function updateListSelection() {
    for (const button of els.episodeList.querySelectorAll('.episode-item')) {
      button.classList.toggle('is-selected', button.dataset.key === state.epKey);
    }
  }

  function currentEpisode() {
    return episodes.find((episode) => episode.key === state.epKey) || null;
  }

  function sceneFor(episode) {
    return (episode && scenes[episode.scene_id]) || {
      scene_id: episode ? episode.scene_id : '',
      rooms: [],
      floors: [],
      voids: [],
      portals: [],
      bounds: null,
      warnings: [],
    };
  }

  // ----------------------------------------------------------------------- //
  // episode selection and panels
  // ----------------------------------------------------------------------- //
  function selectEpisode(key) {
    pausePlayback();
    state.epKey = key;
    const episode = currentEpisode();
    state.selectedEvent = -1;
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    hideTooltip();
    updateListSelection();
    if (!episode) {
      renderNoEpisode();
      draw();
      return;
    }
    state.step = episode.has_trajectory ? episode.trajectory.length - 1 : 0;
    state.floor = 'all';
    renderFloorSelect(episode);
    renderHeader(episode);
    renderMetrics(episode);
    renderFailure(episode);
    renderEvents(episode);
    renderWarnings(episode);
    updatePlayback(episode);
    draw();
  }

  function renderNoEpisode() {
    els.episodeTitle.textContent = 'No episode selected';
    els.episodeSubtitle.textContent = '';
    els.resultBadge.textContent = '—';
    els.resultBadge.className = 'badge badge-empty';
    els.floorSelect.replaceChildren(option('all', 'All floors'));
    els.floorLegend.replaceChildren();
    els.episodeMetrics.replaceChildren();
    els.failureBox.textContent = 'No episode selected.';
    els.eventList.replaceChildren();
    els.eventCount.textContent = '';
    els.warnings.replaceChildren();
    els.warningsPanel.hidden = true;
    els.stepSlider.max = '0';
    els.stepSlider.value = '0';
    updatePlayback(null);
    els.canvasEmpty.hidden = false;
    els.canvasEmpty.textContent = 'No episode matches the current filters';
  }

  function renderHeader(episode) {
    els.episodeTitle.textContent = `${episode.scene_id} / #${episode.episode_id}`;
    const startFloor = episode.start_floor || '?';
    const goalFloor = episode.goal_floor || '?';
    els.episodeSubtitle.textContent = [
      `seed ${episode.seed === null || episode.seed === undefined ? '—' : episode.seed}`,
      `floors ${startFloor} → ${goalFloor}`,
      `${episode.trajectory.length} steps`,
      `${episode.metrics.collision_steps} collision steps`,
    ].join(' · ');
    if (episode.success === true) {
      els.resultBadge.textContent = 'Success';
      els.resultBadge.className = 'badge badge-success';
    } else if (episode.success === false) {
      els.resultBadge.textContent = CATEGORY_LABELS[episode.failure_category]
        || 'Failure';
      els.resultBadge.className = 'badge badge-failure';
    } else {
      els.resultBadge.textContent = 'Unknown result';
      els.resultBadge.className = 'badge badge-empty';
    }
  }

  function metricCard(label, value, detail) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    const name = document.createElement('span');
    name.textContent = label;
    const val = document.createElement('b');
    val.textContent = value;
    card.append(name, val);
    if (detail) {
      const small = document.createElement('small');
      small.textContent = detail;
      card.appendChild(small);
    }
    return card;
  }

  function renderMetrics(episode) {
    const scene = sceneFor(episode);
    const metrics = episode.metrics || {};
    const cards = [
      metricCard(
        'SPL',
        format(episode.spl, 4),
        episode.spl_computed ? 'computed from success + distances' : 'reported'
      ),
      metricCard(
        'Reference distance',
        metres(episode.reference_path && episode.reference_path.distance),
        episode.reference_path && episode.reference_path.distance_computed
          ? 'computed from points' : 'reported'
      ),
      metricCard(
        'Trajectory distance',
        metres(metrics.trajectory_distance),
        `${metrics.steps || 0} steps`
      ),
      metricCard(
        'Distance to goal',
        metres(metrics.final_distance_to_goal),
        'final position → goal'
      ),
      metricCard(
        'Collisions',
        integer(episode.collisions),
        `${metrics.collision_steps || 0} collision steps`
      ),
      metricCard('Timeout', episode.timeout ? 'yes' : 'no'),
      metricCard(
        'Path efficiency',
        metrics.path_efficiency === null || metrics.path_efficiency === undefined
          ? '—' : format(metrics.path_efficiency, 3)
      ),
      metricCard(
        'Floors',
        `${episode.start_floor || '?'} → ${episode.goal_floor || '?'}`,
        `${scene.floors.length} floor records`
      ),
      metricCard(
        'Rooms / voids',
        `${scene.rooms.length} / ${scene.voids.length}`,
        `${scene.portals.length} doors`
      ),
      metricCard(
        'Geometry coverage',
        metricsCoverage(episode),
        'trajectory points near room polygons'
      ),
      metricCard(
        'Yaw frame',
        `${episode.yaw_unit || '?'} / ${episode.yaw_convention || '?'}`,
        yawInferenceDetail(episode)
      ),
      metricCard(
        'Failure type',
        CATEGORY_LABELS[episode.failure_category] || episode.failure_category || '—'
      ),
      metricCard(
        'Seed / episode',
        `${episode.seed === null || episode.seed === undefined ? '—' : episode.seed} / ${episode.episode_id}`
      ),
    ];
    els.episodeMetrics.replaceChildren(...cards);
  }

  function metricsCoverage(episode) {
    return episode.geometry_coverage === null || episode.geometry_coverage === undefined
      ? '—' : `${Math.round(episode.geometry_coverage * 100)}%`;
  }

  function yawInferenceDetail(episode) {
    const inference = episode.yaw_inference || {};
    const samples = inference.movement_samples;
    if (!Number.isFinite(samples) || samples < 1) {
      return 'default frame (no measured motion)';
    }
    return `${samples} motion samples · mean error ${
      format(inference.mean_direction_error_deg, 1)}°`;
  }

  function renderFailure(episode) {
    els.failureBox.replaceChildren();
    const headline = document.createElement('div');
    headline.className = 'failure-headline';
    const category = document.createElement('span');
    category.className = `failure-category cat-${episode.failure_category || 'other'}`;
    category.textContent = CATEGORY_LABELS[episode.failure_category]
      || episode.failure_category || 'Unknown';
    headline.appendChild(category);
    if (episode.timeout) {
      const timeout = document.createElement('span');
      timeout.className = 'muted';
      timeout.textContent = 'timeout';
      headline.appendChild(timeout);
    }
    els.failureBox.appendChild(headline);

    const reason = document.createElement('p');
    reason.className = 'failure-reason';
    reason.textContent = failureText(episode)
      || (episode.success === true ? 'No failure (episode succeeded).'
        : 'No failure_reason supplied.');
    els.failureBox.appendChild(reason);

    const summary = document.createElement('p');
    summary.className = 'failure-summary';
    if (episode.success === true) {
      summary.textContent = [
        `Goal reached with SPL ${format(episode.spl, 4)}.`,
        `Reference ${metres(episode.reference_path.distance)};`,
        `walked ${metres(episode.metrics.trajectory_distance)}.`,
      ].join(' ');
    } else {
      const eventTypes = [...new Set(
        (episode.events || [])
          .filter((event) => event.type !== 'start' && event.type !== 'goal')
          .map((event) => event.type)
      )];
      summary.textContent = [
        eventTypes.length
          ? `Detected evidence: ${eventTypes.join(', ')}.`
          : 'No geometric evidence markers were derived.',
        `Reported collisions ${integer(episode.collisions)};`,
        `final distance to goal ${metres(episode.metrics.final_distance_to_goal)}.`,
      ].join(' ');
    }
    els.failureBox.appendChild(summary);
  }

  function failureText(episode) {
    if (!episode || episode.failure_reason === null || episode.failure_reason === undefined) {
      return '';
    }
    if (typeof episode.failure_reason === 'string') return episode.failure_reason;
    try {
      return JSON.stringify(episode.failure_reason);
    } catch (error) {
      return String(episode.failure_reason);
    }
  }

  function renderEvents(episode) {
    els.eventList.replaceChildren();
    if (!episode) {
      els.eventCount.textContent = '';
      return;
    }
    const events = (episode.events || []).filter(
      (event) => event.type !== 'start' && event.type !== 'goal'
    );
    els.eventCount.textContent = events.length ? `(${events.length})` : '(none)';
    if (!events.length) {
      const note = document.createElement('li');
      note.className = 'empty-note';
      note.textContent = episode.success === true
        ? 'Successful episode; no collision or failure events.'
        : 'No event markers were derived for this episode.';
      els.eventList.appendChild(note);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const event of episode.events) {
      if (event.type === 'start' || event.type === 'goal') continue;
      const index = episode.events.indexOf(event);
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'event-item';
      if (index === state.selectedEvent) button.classList.add('is-selected');

      const dot = document.createElement('span');
      dot.className = 'event-dot';
      dot.style.background = EVENT_COLORS[event.type] || '#9aa7b6';
      const step = document.createElement('span');
      step.className = 'event-step';
      step.textContent = `step ${event.step}`;
      const label = document.createElement('span');
      label.className = 'event-label';
      label.textContent = event.label || event.type;
      const floor = document.createElement('span');
      floor.className = 'event-floor';
      floor.textContent = event.floor_id || '';
      button.append(dot, step, label, floor);
      button.title = `${event.type} at step ${event.step}`;
      button.addEventListener('click', () => focusEvent(episode, index));
      item.appendChild(button);
      fragment.appendChild(item);
    }
    els.eventList.appendChild(fragment);
  }

  function focusEvent(episode, index) {
    const event = episode.events[index];
    if (!event) return;
    state.selectedEvent = index;
    if (
      event.floor_id !== null && event.floor_id !== undefined
      && state.floor !== 'all'
      && String(event.floor_id) !== state.floor
    ) {
      const exists = [...els.floorSelect.options].some(
        (item) => item.value === String(event.floor_id)
      );
      if (exists) {
        state.floor = String(event.floor_id);
        els.floorSelect.value = state.floor;
        resetView();
      }
    }
    state.step = Math.max(0, Math.min(state.stepMax, event.step));
    pausePlayback();
    updatePlayback(episode);
    renderEvents(episode);
    draw();
  }

  function renderWarnings(episode) {
    const warnings = [
      ...(episode ? episode.warnings || [] : []),
      ...(episode ? sceneFor(episode).warnings || [] : []),
    ];
    els.warnings.replaceChildren();
    els.warningsPanel.hidden = warnings.length === 0;
    for (const warning of warnings) {
      const item = document.createElement('li');
      item.textContent = warning;
      els.warnings.appendChild(item);
    }
  }

  function renderRunMeta() {
    const summary = DATA.summary || {};
    const parts = [];
    if (summary.episodes !== undefined) parts.push(`${summary.episodes} episodes`);
    if (summary.scenes !== undefined) parts.push(`${summary.scenes} scenes`);
    if (summary.failures !== undefined) parts.push(`${summary.failures} failures`);
    if (summary.warnings) parts.push(`${summary.warnings} warnings`);
    els.runMeta.textContent = parts.join(' · ') || 'PointNav evaluation';
  }

  function renderGlobalMetrics() {
    const metrics = DATA.metrics || {};
    const computed = (metrics.computed && metrics.computed.overall) || {};
    const provided = metrics.provided || null;
    els.globalMetrics.replaceChildren();
    const title = document.createElement('h2');
    title.textContent = 'Run metrics (computed)';
    els.globalMetrics.appendChild(title);

    addMetricLine('Episodes', integer(computed.episodes));
    addMetricLine('Success rate', percent(computed.success_rate));
    addMetricLine('Mean SPL', format(computed.spl_mean, 4));
    addMetricLine('Mean reference', metres(computed.mean_reference_distance));
    addMetricLine('Collisions', integer(computed.collisions));
    addMetricLine('Timeouts', integer(computed.timeouts));

    if (provided) {
      const flat = provided.flat || {};
      const successRate = firstMetric(flat, ['success_rate', 'success', 'sr']);
      const spl = firstMetric(flat, ['spl', 'mean_spl', 'spl_mean']);
      const count = firstMetric(flat, ['episodes', 'num_episodes', 'count']);
      const extra = [];
      if (successRate !== null) extra.push(`success ${formatNumberPercent(successRate)}`);
      if (spl !== null) extra.push(`SPL ${format(spl, 3)}`);
      if (count !== null) extra.push(`${integer(count)} episodes`);
      const line = document.createElement('div');
      line.className = 'metric-line';
      const label = document.createElement('span');
      label.textContent = 'metrics.json';
      const value = document.createElement('b');
      value.textContent = extra.length ? extra.join(' · ') : 'loaded';
      line.append(label, value);
      els.globalMetrics.appendChild(line);
      if (provided.source) {
        const source = document.createElement('div');
        source.className = 'metric-line';
        const sourceLabel = document.createElement('span');
        sourceLabel.textContent = 'source';
        const sourceValue = document.createElement('b');
        sourceValue.textContent = String(provided.source).split('/').pop();
        source.append(sourceLabel, sourceValue);
        els.globalMetrics.appendChild(source);
      }
    }
  }

  function addMetricLine(labelText, valueText) {
    const line = document.createElement('div');
    line.className = 'metric-line';
    const label = document.createElement('span');
    label.textContent = labelText;
    const value = document.createElement('b');
    value.textContent = valueText;
    line.append(label, value);
    els.globalMetrics.appendChild(line);
  }

  function firstMetric(flat, names) {
    for (const name of names) {
      for (const [key, value] of Object.entries(flat || {})) {
        const lower = key.toLowerCase();
        if (lower === name || lower.endsWith(`.${name}`)) {
          return typeof value === 'number' ? value : null;
        }
      }
    }
    return null;
  }

  function renderFooter() {
    const summary = DATA.summary || {};
    const source = DATA.source || {};
    const warnings = DATA.warnings || [];
    const parts = [
      `replay v${DATA.version || 1}`,
      DATA.generated_utc ? `generated ${DATA.generated_utc}` : '',
      source.episodes ? `source ${String(source.episodes).split('/').pop()}` : '',
      summary.scenes !== undefined ? `${summary.scenes} scenes` : '',
      warnings.length ? `${warnings.length} build warnings` : 'no build warnings',
    ].filter(Boolean);
    els.footerInfo.textContent = parts.join(' · ');
    els.footerInfo.title = warnings.slice(0, 50).join('\n')
      + (warnings.length > 50 ? `\n... ${warnings.length - 50} more` : '');
    els.footerHint.textContent =
      'Space play/pause · ←/→ step · Home/End · double-click fit';
  }

  // ----------------------------------------------------------------------- //
  // floors
  // ----------------------------------------------------------------------- //
  function floorIdsForEpisode(episode) {
    const ids = [];
    const add = (value) => {
      if (value === null || value === undefined) return;
      const text = String(value);
      if (!ids.includes(text)) ids.push(text);
    };
    for (const floor of sceneFor(episode).floors || []) add(floor.id);
    add(episode.start_floor);
    add(episode.goal_floor);
    for (const step of episode.trajectory || []) add(step.floor_id);
    for (const id of (episode.reference_path && episode.reference_path.floor_ids) || []) {
      add(id);
    }
    return ids;
  }

  function floorRecord(scene, id) {
    return (scene.floors || []).find((floor) => String(floor.id) === String(id))
      || null;
  }

  function floorColor(scene, id) {
    const index = (scene.floors || []).findIndex(
      (floor) => String(floor.id) === String(id)
    );
    return FLOOR_COLORS[(index < 0 ? 0 : index) % FLOOR_COLORS.length];
  }

  function renderFloorSelect(episode) {
    const scene = sceneFor(episode);
    const ids = floorIdsForEpisode(episode);
    els.floorSelect.replaceChildren(option('all', 'All floors'));
    els.floorLegend.replaceChildren();
    for (const id of ids) {
      const record = floorRecord(scene, id);
      const elevation = record && Number.isFinite(record.elevation)
        ? ` · ${format(record.elevation, 2)} m` : '';
      els.floorSelect.appendChild(option(id, `${id}${elevation}`));
      const chip = document.createElement('span');
      chip.className = 'floor-chip';
      const swatch = document.createElement('i');
      swatch.style.background = floorColor(scene, id);
      const text = document.createElement('span');
      text.textContent = id;
      chip.append(swatch, text);
      els.floorLegend.appendChild(chip);
    }
    if (!ids.length) {
      els.floorLegend.textContent = 'No floor metadata';
    }
  }

  function floorVisible(floorId) {
    if (state.floor === 'all') return true;
    if (floorId === null || floorId === undefined) return true;
    return String(floorId) === state.floor;
  }

  // ----------------------------------------------------------------------- //
  // playback
  // ----------------------------------------------------------------------- //
  function updatePlayback(episode) {
    const steps = episode && episode.has_trajectory
      ? episode.trajectory.length : 0;
    const max = Math.max(0, steps - 1);
    state.stepMax = max;
    state.step = Math.max(0, Math.min(state.step, max));
    els.stepSlider.max = String(max);
    els.stepSlider.value = String(state.step);
    els.stepSlider.disabled = steps < 2;
    els.playButton.disabled = steps < 1;
    els.stepBack.disabled = steps < 1 || state.step <= 0;
    els.stepForward.disabled = steps < 1 || state.step >= max;
    els.restartButton.disabled = steps < 1 || state.step <= 0;
    els.endButton.disabled = steps < 1 || state.step >= max;
    els.playButton.textContent = state.playing
      ? 'Pause'
      : (steps > 1 && state.step >= max ? 'Replay' : 'Play');

    if (!episode) {
      els.stepLabel.textContent = 'step —';
      els.actionLabel.textContent = '';
      return;
    }
    if (!steps) {
      els.stepLabel.textContent = 'no trajectory recorded';
      els.actionLabel.textContent = 'reference path only';
      return;
    }
    const step = episode.trajectory[Math.min(state.step, max)];
    els.stepLabel.textContent = `step ${state.step} / ${max}`;
    const parts = [];
    if (step.action !== null && step.action !== undefined) {
      parts.push(`action ${String(step.action)}`);
    }
    const yawDegreesValue = yawDegrees(episode, step.yaw);
    if (yawDegreesValue !== null) {
      parts.push(`yaw ${format(yawDegreesValue, 1)}°`);
    }
    if (step.floor_id) parts.push(`floor ${step.floor_id}`);
    if (step.collision) parts.push('collision');
    els.actionLabel.textContent = parts.join(' · ');
  }

  function setStep(next) {
    const episode = currentEpisode();
    if (!episode || !episode.has_trajectory) return;
    const max = Math.max(0, episode.trajectory.length - 1);
    state.step = Math.max(0, Math.min(max, next));
    if (state.playing && state.step >= max) pausePlayback();
    else updatePlayback(episode);
    draw();
  }

  function togglePlayback() {
    if (state.playing) pausePlayback();
    else startPlayback();
  }

  function startPlayback() {
    const episode = currentEpisode();
    if (!episode || !episode.has_trajectory) return;
    const max = Math.max(0, episode.trajectory.length - 1);
    if (state.step >= max) state.step = 0;
    state.playing = true;
    state.lastTs = 0;
    state.acc = 0;
    updatePlayback(episode);
    draw();
    cancelAnimationFrame(rafHandle);
    rafHandle = requestAnimationFrame(tickPlayback);
  }

  function tickPlayback(timestamp) {
    if (!state.playing) return;
    const episode = currentEpisode();
    if (!episode || !episode.has_trajectory) {
      pausePlayback();
      return;
    }
    const max = episode.trajectory.length - 1;
    if (!state.lastTs) state.lastTs = timestamp;
    const dt = Math.min(150, Math.max(0, timestamp - state.lastTs));
    state.lastTs = timestamp;
    state.acc += dt;
    const interval = 240 / Math.max(0.1, state.speed);
    let changed = false;
    while (state.acc >= interval) {
      state.acc -= interval;
      if (state.step < max) {
        state.step += 1;
        changed = true;
      } else {
        pausePlayback();
        return;
      }
    }
    if (changed) {
      updatePlayback(episode);
      draw();
    }
    rafHandle = requestAnimationFrame(tickPlayback);
  }

  function pausePlayback() {
    state.playing = false;
    cancelAnimationFrame(rafHandle);
    rafHandle = 0;
    if (els && els.playButton) updatePlayback(currentEpisode());
  }

  function yawDegrees(episode, value) {
    if (!Number.isFinite(value)) return null;
    return episode && episode.yaw_unit === 'degrees'
      ? value
      : value * 180 / Math.PI;
  }

  // ----------------------------------------------------------------------- //
  // map geometry and drawing
  // ----------------------------------------------------------------------- //
  function resizeCanvas() {
    if (!els.canvasWrap) return false;
    const rect = els.canvasWrap.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (width === state.width && height === state.height) return false;
    state.width = width;
    state.height = height;
    dpr = window.devicePixelRatio || 1;
    els.canvas.width = Math.max(1, Math.round(width * dpr));
    els.canvas.height = Math.max(1, Math.round(height * dpr));
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  }

  function resetView() {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
  }

  function includeFloor(floorId) {
    if (state.floor === 'all') return true;
    if (floorId === null || floorId === undefined) return true;
    return String(floorId) === state.floor;
  }

  function computeView(episode) {
    const scene = sceneFor(episode);
    let minX = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxZ = -Infinity;

    const addPoint = (point) => {
      if (!Array.isArray(point) || point.length < 3) return;
      const x = point[0];
      const z = point[2];
      if (!Number.isFinite(x) || !Number.isFinite(z)) return;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    };
    const addPolygon = (polygon) => {
      for (const point of polygon || []) addPoint([point[0], 0, point[1]]);
    };

    for (const room of scene.rooms || []) {
      if (includeFloor(room.floor_id)) addPolygon(room.polygon);
    }
    for (const voidPolygon of scene.voids || []) {
      if (includeFloor(voidPolygon.floor_id)) addPolygon(voidPolygon.polygon);
    }
    const reference = episode.reference_path || { points: [], floor_ids: [] };
    reference.points.forEach((point, index) => {
      if (includeFloor((reference.floor_ids || [])[index])) addPoint(point);
    });
    for (const step of episode.trajectory || []) {
      if (includeFloor(step.floor_id)) addPoint(step.position);
    }
    if (includeFloor(episode.start_floor)) addPoint(episode.start);
    if (includeFloor(episode.goal_floor)) addPoint(episode.goal);

    if (!Number.isFinite(minX) && scene.bounds) {
      minX = scene.bounds[0];
      minZ = scene.bounds[1];
      maxX = scene.bounds[2];
      maxZ = scene.bounds[3];
    }
    if (!Number.isFinite(minX)) {
      minX = -3; maxX = 3; minZ = -3; maxZ = 3;
    }
    let spanX = maxX - minX;
    let spanZ = maxZ - minZ;
    if (spanX < 1) {
      const centre = (minX + maxX) / 2;
      minX = centre - 0.5; maxX = centre + 0.5; spanX = 1;
    }
    if (spanZ < 1) {
      const centre = (minZ + maxZ) / 2;
      minZ = centre - 0.5; maxZ = centre + 0.5; spanZ = 1;
    }
    const pad = Math.max(0.5, Math.max(spanX, spanZ) * 0.07);
    minX -= pad; maxX += pad; minZ -= pad; maxZ += pad;
    spanX = maxX - minX;
    spanZ = maxZ - minZ;

    const width = state.width;
    const height = state.height;
    const margin = 24;
    const baseScale = Math.min(
      (width - margin * 2) / spanX,
      (height - margin * 2) / spanZ
    );
    const scale = Math.max(0.0001, baseScale * state.zoom);
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    const sx = (x) => width / 2 + state.panX + (x - cx) * scale;
    const sy = (z) => height / 2 + state.panY + (z - cz) * scale;
    const wx = (px) => cx + (px - width / 2 - state.panX) / scale;
    const wz = (py) => cz + (py - height / 2 - state.panY) / scale;
    return { minX, maxX, minZ, maxZ, spanX, spanZ, scale, cx, cz, width, height, sx, sy, wx, wz };
  }

  function draw() {
    if (!ctx || !els) return;
    resizeCanvas();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.fillStyle = '#0a0f15';
    ctx.fillRect(0, 0, state.width, state.height);
    state.hitMarkers = [];

    const episode = currentEpisode();
    if (!episode) {
      els.canvasEmpty.hidden = false;
      return;
    }
    els.canvasEmpty.hidden = true;
    const scene = sceneFor(episode);
    const view = computeView(episode);
    drawGrid(view);
    drawRooms(view, scene);
    drawReferencePath(view, episode);
    drawTrajectory(view, episode);
    drawEndpoints(view, episode);
    drawEvents(view, episode);
    drawCurrent(view, episode);
    drawCompassAndScale(view);
    if (state.floor !== 'all') {
      drawNotice(`Floor ${state.floor}`, 26);
    }
    if (!scene.rooms.length) {
      drawNotice('No room polygons for this scene', 44);
    }
    if (!episode.has_trajectory) {
      drawNotice('No trajectory recorded · reference path only', 62);
    }
  }

  function drawGrid(view) {
    const target = 74;
    const step = niceStep(target / view.scale);
    const left = view.wx(0);
    const right = view.wx(view.width);
    const top = view.wz(0);
    const bottom = view.wz(view.height);
    ctx.save();
    ctx.lineWidth = 1;
    for (let x = Math.floor(left / step) * step; x <= right; x += step) {
      const major = Math.abs(x / (step * 5) - Math.round(x / (step * 5))) < 1e-6;
      ctx.strokeStyle = major
        ? 'rgba(160, 180, 205, 0.10)' : 'rgba(160, 180, 205, 0.045)';
      ctx.beginPath();
      ctx.moveTo(view.sx(x), 0);
      ctx.lineTo(view.sx(x), view.height);
      ctx.stroke();
    }
    for (let z = Math.floor(top / step) * step; z <= bottom; z += step) {
      const major = Math.abs(z / (step * 5) - Math.round(z / (step * 5))) < 1e-6;
      ctx.strokeStyle = major
        ? 'rgba(160, 180, 205, 0.10)' : 'rgba(160, 180, 205, 0.045)';
      ctx.beginPath();
      ctx.moveTo(0, view.sy(z));
      ctx.lineTo(view.width, view.sy(z));
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(154, 167, 182, 0.55)';
    ctx.font = '9px "Inter", "Segoe UI", sans-serif';
    ctx.fillText('x', view.width - 14, view.height - 9);
    ctx.fillText('z', 8, 14);
    ctx.restore();
  }

  function niceStep(target) {
    const powers = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
    for (const value of powers) {
      if (value >= target) return value;
    }
    return 200;
  }

  function drawRooms(view, scene) {
    const floors = (scene.floors || []).slice().sort((a, b) => {
      const ae = Number.isFinite(a.elevation) ? a.elevation : 0;
      const be = Number.isFinite(b.elevation) ? b.elevation : 0;
      return ae - be;
    });
    const orderedIds = floors.map((floor) => String(floor.id));
    for (const room of scene.rooms || []) {
      if (!includeFloor(room.floor_id)) continue;
      const id = String(room.floor_id);
      if (!orderedIds.includes(id)) orderedIds.push(id);
    }
    for (const voidPolygon of scene.voids || []) {
      if (!includeFloor(voidPolygon.floor_id)) continue;
      const id = String(voidPolygon.floor_id);
      if (id !== 'null' && !orderedIds.includes(id)) orderedIds.push(id);
    }
    if (state.floor !== 'all' && !orderedIds.includes(state.floor)) {
      orderedIds.push(state.floor);
    }

    for (const floorId of orderedIds) {
      if (!includeFloor(floorId)) continue;
      const color = floorColor(scene, floorId);
      const rooms = (scene.rooms || []).filter(
        (room) => String(room.floor_id) === floorId
      );
      for (const room of rooms) {
        drawPolygon(view, room.polygon);
        ctx.fillStyle = colourWithAlpha(color, state.floor === 'all' ? 0.10 : 0.17);
        ctx.fill();
        ctx.strokeStyle = colourWithAlpha(color, 0.58);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        drawRoomLabel(view, room, color);
      }
    }

    for (const voidPolygon of scene.voids || []) {
      if (!includeFloor(voidPolygon.floor_id)) continue;
      drawPolygon(view, voidPolygon.polygon);
      ctx.fillStyle = 'rgba(255, 180, 84, 0.12)';
      ctx.fill();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = 'rgba(255, 180, 84, 0.8)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.setLineDash([]);
      const centre = polygonCentre(voidPolygon.polygon);
      if (centre && polygonScreenArea(view, voidPolygon.polygon) > 300) {
        ctx.fillStyle = 'rgba(255, 180, 84, 0.9)';
        ctx.font = '9px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('void', view.sx(centre[0]), view.sy(centre[1]) + 3);
        ctx.textAlign = 'left';
      }
    }

    for (const portal of scene.portals || []) {
      if (!includeFloor(portal.floor_id)) continue;
      const x = view.sx(portal.position[0]);
      const y = view.sy(portal.position[1]);
      const radius = Math.max(4, Math.min(16, portal.radius * view.scale));
      ctx.save();
      ctx.strokeStyle = 'rgba(93, 211, 158, 0.7)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(93, 211, 158, 0.85)';
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function drawRoomLabel(view, room, color) {
    const centre = room.center || polygonCentre(room.polygon);
    if (!centre) return;
    const area = polygonScreenArea(view, room.polygon);
    if (area < 650) return;
    const label = room.type ? `${room.id} · ${room.type}` : String(room.id);
    ctx.save();
    ctx.fillStyle = colourWithAlpha(color, 0.92);
    ctx.font = '10px "Inter", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, view.sx(centre[0]), view.sy(centre[1]));
    ctx.restore();
  }

  function drawPolygon(view, polygon) {
    if (!polygon || polygon.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(view.sx(polygon[0][0]), view.sy(polygon[0][1]));
    for (let index = 1; index < polygon.length; index += 1) {
      ctx.lineTo(view.sx(polygon[index][0]), view.sy(polygon[index][1]));
    }
    ctx.closePath();
  }

  function polygonCentre(polygon) {
    if (!polygon || !polygon.length) return null;
    let x = 0;
    let z = 0;
    for (const point of polygon) {
      x += point[0];
      z += point[1];
    }
    return [x / polygon.length, z / polygon.length];
  }

  function polygonScreenArea(view, polygon) {
    if (!polygon || polygon.length < 3) return 0;
    let area = 0;
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const next = polygon[(index + 1) % polygon.length];
      area += view.sx(current[0]) * view.sy(next[1])
        - view.sx(next[0]) * view.sy(current[1]);
    }
    return Math.abs(area / 2);
  }

  function drawReferencePath(view, episode) {
    const reference = episode.reference_path || { points: [], floor_ids: [] };
    const points = reference.points || [];
    if (points.length < 2) return;
    const floorIds = reference.floor_ids || [];
    ctx.save();
    ctx.strokeStyle = 'rgba(111, 178, 255, 0.9)';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([7, 5]);
    for (let index = 1; index < points.length; index += 1) {
      if (
        state.floor !== 'all'
        && !floorVisible(floorIds[index - 1])
        && !floorVisible(floorIds[index])
      ) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(view.sx(points[index - 1][0]), view.sy(points[index - 1][2]));
      ctx.lineTo(view.sx(points[index][0]), view.sy(points[index][2]));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const point of points) {
      ctx.beginPath();
      ctx.arc(view.sx(point[0]), view.sy(point[2]), 2.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(111, 178, 255, 0.85)';
      ctx.fill();
    }
    ctx.restore();
  }

  function drawTrajectory(view, episode) {
    const trajectory = episode.trajectory || [];
    if (!trajectory.length) return;
    const colour = episode.success === true ? '#5dd39e' : '#ffb454';

    // Ghost of the full path so the playback head has context.
    if (trajectory.length > 1) {
      ctx.save();
      ctx.strokeStyle = colourWithAlpha(colour, 0.16);
      ctx.lineWidth = 2;
      for (let index = 1; index < trajectory.length; index += 1) {
        const previous = trajectory[index - 1];
        const current = trajectory[index];
        if (
          state.floor !== 'all'
          && !floorVisible(previous.floor_id)
          && !floorVisible(current.floor_id)
        ) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(
          view.sx(previous.position[0]), view.sy(previous.position[2])
        );
        ctx.lineTo(
          view.sx(current.position[0]), view.sy(current.position[2])
        );
        ctx.stroke();
      }
      ctx.restore();
    }

    const upto = Math.max(0, Math.min(state.step, trajectory.length - 1));
    if (upto > 0) {
      ctx.save();
      for (let index = 1; index <= upto; index += 1) {
        const previous = trajectory[index - 1];
        const current = trajectory[index];
        if (
          state.floor !== 'all'
          && !floorVisible(previous.floor_id)
          && !floorVisible(current.floor_id)
        ) {
          continue;
        }
        ctx.strokeStyle = colourWithAlpha(colour, 0.95);
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(view.sx(previous.position[0]), view.sy(previous.position[2]));
        ctx.lineTo(view.sx(current.position[0]), view.sy(current.position[2]));
        ctx.stroke();
        if (current.collision) {
          ctx.strokeStyle = 'rgba(255, 80, 80, 0.95)';
          ctx.lineWidth = 4.2;
          ctx.beginPath();
          ctx.moveTo(view.sx(previous.position[0]), view.sy(previous.position[2]));
          ctx.lineTo(view.sx(current.position[0]), view.sy(current.position[2]));
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    const stride = Math.max(1, Math.ceil(trajectory.length / 80));
    ctx.save();
    ctx.fillStyle = colourWithAlpha(colour, 0.55);
    for (let index = 0; index <= upto; index += stride) {
      const step = trajectory[index];
      if (!floorVisible(step.floor_id)) continue;
      ctx.beginPath();
      ctx.arc(view.sx(step.position[0]), view.sy(step.position[2]), 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawEndpoints(view, episode) {
    if (episode.start) {
      const visible = floorVisible(episode.start_floor);
      drawEndpoint(view, episode.start, 'start', episode.start_floor, visible);
    }
    if (episode.goal) {
      const visible = floorVisible(episode.goal_floor);
      drawEndpoint(view, episode.goal, 'goal', episode.goal_floor, visible);
    }
  }

  function drawEndpoint(view, position, kind, floorId, visible) {
    const x = view.sx(position[0]);
    const y = view.sy(position[2]);
    const color = kind === 'start' ? '#5dd39e' : '#ff7b72';
    ctx.save();
    ctx.globalAlpha = visible ? 1 : 0.28;
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#071016';
    ctx.fillStyle = color;
    ctx.beginPath();
    if (kind === 'start') {
      ctx.arc(x, y, 7, 0, Math.PI * 2);
    } else {
      const radius = 8;
      ctx.moveTo(x, y - radius);
      ctx.lineTo(x + radius, y);
      ctx.lineTo(x, y + radius);
      ctx.lineTo(x - radius, y);
      ctx.closePath();
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#071016';
    ctx.font = 'bold 9px "Inter", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(kind === 'start' ? 'S' : 'G', x, y + 0.5);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '10px "Inter", "Segoe UI", sans-serif';
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(7, 16, 22, 0.85)';
    const label = `${kind}${floorId ? ` · ${floorId}` : ''}`;
    ctx.strokeText(label, x + 11, y - 8);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 11, y - 8);
    ctx.restore();
    state.hitMarkers.push({
      x,
      y,
      title: `${kind}${floorId ? ` · floor ${floorId}` : ''}`,
      detail: position.map((value) => format(value, 2)).join(', '),
    });
  }

  function drawEvents(view, episode) {
    for (let index = 0; index < (episode.events || []).length; index += 1) {
      const event = episode.events[index];
      if (event.type === 'start' || event.type === 'goal') continue;
      if (!floorVisible(event.floor_id)) continue;
      const x = view.sx(event.position[0]);
      const y = view.sy(event.position[2]);
      const color = EVENT_COLORS[event.type] || '#9aa7b6';
      const alpha = event.step <= state.step ? 1 : 0.35;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = colourWithAlpha(color, 0.22);
      if (event.type === 'collision') {
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4);
        ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x + 4, y - 4);
        ctx.lineTo(x - 4, y + 4);
        ctx.stroke();
      } else if (event.type === 'stuck_door' || event.type === 'stuck') {
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = 'bold 10px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('!', x, y + 0.5);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      } else if (event.type === 'through_wall') {
        ctx.beginPath();
        ctx.moveTo(x, y - 8);
        ctx.lineTo(x + 8, y + 6);
        ctx.lineTo(x - 8, y + 6);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (event.type === 'cross_void') {
        ctx.beginPath();
        ctx.rect(x - 7, y - 7, 14, 14);
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - 4, y - 4);
        ctx.lineTo(x + 4, y + 4);
        ctx.moveTo(x + 4, y - 4);
        ctx.lineTo(x - 4, y + 4);
        ctx.stroke();
      } else if (event.type === 'floor_jump') {
        ctx.beginPath();
        ctx.moveTo(x, y - 9);
        ctx.lineTo(x + 8, y);
        ctx.lineTo(x, y + 9);
        ctx.lineTo(x - 8, y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = 'bold 9px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('↕', x, y + 0.5);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      } else {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      if (index === state.selectedEvent) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 13, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
      state.hitMarkers.push({
        x,
        y,
        title: event.label || event.type,
        detail: `step ${event.step}${event.floor_id ? ` · floor ${event.floor_id}` : ''} · ${event.type}`,
        eventIndex: index,
      });
    }
  }

  function drawCurrent(view, episode) {
    const trajectory = episode.trajectory || [];
    if (!trajectory.length) return;
    const index = Math.max(0, Math.min(state.step, trajectory.length - 1));
    const step = trajectory[index];
    if (!floorVisible(step.floor_id)) return;
    const x = view.sx(step.position[0]);
    const y = view.sy(step.position[2]);
    let dx = null;
    let dz = null;
    if (Number.isFinite(step.yaw)) {
      const angle = episode.yaw_unit === 'degrees'
        ? step.yaw * Math.PI / 180 : step.yaw;
      if (episode.yaw_convention === 'habitat') {
        dx = -Math.sin(angle);
        dz = -Math.cos(angle);
      } else {
        dx = Math.cos(angle);
        dz = -Math.sin(angle);
      }
    } else if (index > 0) {
      const previous = trajectory[index - 1].position;
      dx = step.position[0] - previous[0];
      dz = step.position[2] - previous[2];
    }
    if (dx !== null && dz !== null) {
      const length = Math.hypot(dx, dz) || 1;
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 209, 102, 0.95)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (dx / length) * 17, y + (dz / length) * 17);
      ctx.stroke();
      ctx.restore();
    }
    ctx.save();
    ctx.fillStyle = '#ffd166';
    ctx.strokeStyle = '#071016';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, 9.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawCompassAndScale(view) {
    ctx.save();
    ctx.strokeStyle = 'rgba(210, 220, 235, 0.7)';
    ctx.fillStyle = 'rgba(210, 220, 235, 0.7)';
    ctx.lineWidth = 1.4;
    const ax = view.width - 26;
    const ay = 24;
    ctx.beginPath();
    ctx.moveTo(ax, ay + 12);
    ctx.lineTo(ax, ay - 12);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ax, ay - 16);
    ctx.lineTo(ax - 4, ay - 9);
    ctx.lineTo(ax + 4, ay - 9);
    ctx.closePath();
    ctx.fill();
    ctx.font = '9px "Inter", "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('+x', ax + 14, ay + 3);
    ctx.textAlign = 'left';
    const text = '-z';
    ctx.fillText(text, ax - 24, ay + 3);

    let metres = 1;
    for (const candidate of [0.5, 1, 2, 5, 10, 20, 50]) {
      if (candidate * view.scale >= 52) {
        metres = candidate;
        break;
      }
    }
    const length = metres * view.scale;
    const bx = 14;
    const by = view.height - 16;
    ctx.strokeStyle = 'rgba(210, 220, 235, 0.65)';
    ctx.beginPath();
    ctx.moveTo(bx, by - 5);
    ctx.lineTo(bx, by);
    ctx.lineTo(bx + length, by);
    ctx.lineTo(bx + length, by - 5);
    ctx.stroke();
    ctx.fillStyle = 'rgba(210, 220, 235, 0.75)';
    ctx.font = '9px "Inter", "Segoe UI", sans-serif';
    ctx.fillText(`${metres} m`, bx + length + 5, by);
    ctx.restore();
  }

  function drawNotice(text, y) {
    ctx.save();
    ctx.fillStyle = 'rgba(255, 180, 84, 0.92)';
    ctx.font = '11px "Inter", "Segoe UI", sans-serif';
    ctx.fillText(text, 12, y);
    ctx.restore();
  }

  function colourWithAlpha(hex, alpha) {
    const value = hex.replace('#', '');
    const number = parseInt(value.length === 3
      ? value.split('').map((part) => part + part).join('')
      : value, 16);
    const red = (number >> 16) & 255;
    const green = (number >> 8) & 255;
    const blue = number & 255;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  // ----------------------------------------------------------------------- //
  // pointer interaction
  // ----------------------------------------------------------------------- //
  function canvasPoint(event) {
    const rect = els.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function onWheel(event) {
    const episode = currentEpisode();
    if (!episode) return;
    event.preventDefault();
    const point = canvasPoint(event);
    const view = computeView(episode);
    const worldX = view.wx(point.x);
    const worldZ = view.wz(point.y);
    const factor = Math.exp(-event.deltaY * 0.0015);
    state.zoom = Math.max(0.25, Math.min(12, state.zoom * factor));
    const next = computeView(episode);
    state.panX += point.x - next.sx(worldX);
    state.panY += point.y - next.sy(worldZ);
    draw();
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    const point = canvasPoint(event);
    state.drag = {
      x: point.x,
      y: point.y,
      panX: state.panX,
      panY: state.panY,
      moved: false,
    };
    els.canvas.setPointerCapture(event.pointerId);
    els.canvasWrap.classList.add('is-dragging');
  }

  function onPointerMove(event) {
    const point = canvasPoint(event);
    if (state.drag) {
      const dx = point.x - state.drag.x;
      const dy = point.y - state.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) state.drag.moved = true;
      state.panX = state.drag.panX + dx;
      state.panY = state.drag.panY + dy;
      hideTooltip();
      draw();
      return;
    }
    const marker = nearestMarker(point);
    if (marker) {
      showTooltip(marker, event.clientX, event.clientY);
    } else {
      hideTooltip();
    }
  }

  function onPointerUp(event) {
    if (!state.drag) return;
    const moved = state.drag.moved;
    state.drag = null;
    els.canvasWrap.classList.remove('is-dragging');
    if (els.canvas.hasPointerCapture && els.canvas.hasPointerCapture(event.pointerId)) {
      els.canvas.releasePointerCapture(event.pointerId);
    }
    if (!moved) {
      const point = canvasPoint(event);
      const marker = nearestMarker(point);
      if (marker && marker.eventIndex !== undefined) {
        focusEvent(currentEpisode(), marker.eventIndex);
      }
    }
  }

  function nearestMarker(point) {
    let best = null;
    let bestDistance = 15 * 15;
    for (const marker of state.hitMarkers) {
      const dx = marker.x - point.x;
      const dy = marker.y - point.y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        best = marker;
        bestDistance = distance;
      }
    }
    return best;
  }

  function showTooltip(marker, clientX, clientY) {
    const rect = els.canvasWrap.getBoundingClientRect();
    els.tooltip.replaceChildren();
    const title = document.createElement('b');
    title.textContent = marker.title;
    const detail = document.createElement('span');
    detail.textContent = marker.detail;
    els.tooltip.append(title, detail);
    els.tooltip.hidden = false;
    let left = clientX - rect.left + 13;
    let top = clientY - rect.top + 13;
    const maxLeft = rect.width - els.tooltip.offsetWidth - 8;
    const maxTop = rect.height - els.tooltip.offsetHeight - 8;
    left = Math.max(6, Math.min(maxLeft, left));
    top = Math.max(6, Math.min(maxTop, top));
    els.tooltip.style.left = `${left}px`;
    els.tooltip.style.top = `${top}px`;
  }

  function hideTooltip() {
    if (els && els.tooltip) els.tooltip.hidden = true;
  }

  // ----------------------------------------------------------------------- //
  // formatting
  // ----------------------------------------------------------------------- //
  function format(value, digits = 3) {
    return Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
  }

  function metres(value) {
    return Number.isFinite(value) ? `${Number(value).toFixed(2)} m` : '—';
  }

  function integer(value) {
    return Number.isFinite(value) ? String(Math.round(value)) : '—';
  }

  function percent(value) {
    return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
  }

  function formatNumberPercent(value) {
    return Number.isFinite(value)
      ? `${(value <= 1 ? value * 100 : value).toFixed(1)}%`
      : '—';
  }

  init();
})();
