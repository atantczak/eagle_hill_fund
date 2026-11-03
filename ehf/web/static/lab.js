/* Eagle Hill lab: the browser, run pages, playground and symbol charts.
   Chart pages load TradingView's lightweight-charts (v5) before this file. */
window.EHF = (function () {
  'use strict';

  // ---- small helpers ------------------------------------------------------------
  const $ = (sel, root) => (root || document).querySelector(sel);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const theme = () => ({ text: css('--muted'), grid: css('--grid'), accent: css('--accent'), blue: css('--c-blue'), orange: css('--c-orange'), aqua: css('--c-aqua'), yellow: css('--c-yellow'), magenta: css('--c-magenta'), grey: css('--c-grey'), up: css('--up'), down: css('--down'), mono: css('--mono') });
  const onTheme = (cb) => { document.addEventListener('ehf:theme', cb); matchMedia('(prefers-color-scheme: dark)').addEventListener('change', cb); };
  const LW = () => window.LightweightCharts;
  const last = (arr) => { if (!arr) return null; for (let i = arr.length - 1; i >= 0; i--) if (arr[i] != null) return arr[i]; return null; };
  const alpha = (hex, a) => { const h = hex.replace('#', ''); const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; };
  const fin = (v) => v != null && isFinite(v);
  const fmt = {
    pct: (v, d = 1) => (fin(v) ? (v * 100).toFixed(d) + '%' : '–'),
    sign: (v, d = 1) => (fin(v) ? (v > 0 ? '+' : '') + (v * 100).toFixed(d) + '%' : '–'),
    num: (v, d = 2) => (fin(v) ? Number(v).toFixed(d) : '–'),
    int: (v) => (fin(v) ? Math.round(v).toLocaleString() : '–'),
    money: (v) => (fin(v) ? '$' + Math.round(v).toLocaleString() : '–'),
    px: (v) => (fin(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(2)) : '–'),
    vol: (v) => (fin(v) ? (v >= 1e9 ? (v / 1e9).toFixed(2) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : String(Math.round(v))) : '–'),
    date: (s) => s || '',
  };
  const signCls = (v) => (!fin(v) ? '' : v >= 0 ? 'pos' : 'neg');
  const HIDDEN = new Set(['vix', 'vix3m', 'vvix', 'skew', 'cash', 'base', 'benchmark']);
  const scalar = (v) => v == null || typeof v !== 'object';
  const paramChips = (params) => Object.entries(params || {}).filter(([k, v]) => !HIDDEN.has(k) && scalar(v)).map(([k, v]) => `<span class="chip">${esc(k)}=<b>${esc(v)}</b></span>`).join('');
  const paramsShort = (params) => Object.entries(params || {}).filter(([k, v]) => !HIDDEN.has(k) && k !== 'asset' && scalar(v)).map(([k, v]) => `${k}=${v}`).join(' ');
  const costsText = (c) => { c = c || {}; const bits = [`slippage ${c.slippage_bps ?? 0} bps`]; if (c.commission_per_share) bits.push(`$${c.commission_per_share}/share commission`); if (c.borrow_bps_annual) bits.push(`${c.borrow_bps_annual} bps borrow`); if (c.frictionless) bits.push('FRICTIONLESS (debug)'); return bits.join(', '); };
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  async function getJSON(url, opts) { const r = await fetch(url, opts); if (!r.ok) { let msg = r.statusText; try { msg = (await r.json()).detail || msg; } catch (e) { /* plain */ } throw new Error(msg); } return r.json(); }

  // ---- chart primitives ---------------------------------------------------------
  function makeChart(container) {
    const t = theme();
    const chart = LW().createChart(container, {
      autoSize: true,
      layout: { background: { type: 'solid', color: 'transparent' }, textColor: t.text, fontFamily: t.mono, fontSize: 11, attributionLogo: true, panes: { separatorColor: t.grid, separatorHoverColor: alpha(t.blue, 0.2), enableResize: true } },
      grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } },
      rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.08, bottom: 0.08 } },
      timeScale: { borderVisible: false, rightOffset: 3, minBarSpacing: 0.15 },
      crosshair: { mode: LW().CrosshairMode.Normal, vertLine: { labelBackgroundColor: t.accent, color: alpha(t.grey, 0.5) }, horzLine: { labelBackgroundColor: t.accent, color: alpha(t.grey, 0.5) } },
    });
    onTheme(() => {
      const t = theme();
      chart.applyOptions({ layout: { textColor: t.text, panes: { separatorColor: t.grid } }, grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } }, crosshair: { vertLine: { labelBackgroundColor: t.accent }, horzLine: { labelBackgroundColor: t.accent } } });
    });
    return chart;
  }
  function priceFormat(o) { return o.format ? { type: 'custom', formatter: o.format, minMove: o.minMove || 0.01 } : (o.priceFormat || { type: 'price', precision: 2, minMove: 0.01 }); }
  function addLine(chart, data, o = {}) {
    const opts = { color: o.color, lineWidth: o.width || 2, priceLineVisible: false, lastValueVisible: o.last !== false, crosshairMarkerVisible: true, crosshairMarkerRadius: 4, lineType: o.step ? LW().LineType.WithSteps : LW().LineType.Simple, lineStyle: o.dashed ? LW().LineStyle.Dashed : LW().LineStyle.Solid, priceFormat: priceFormat(o) };
    const s = chart.addSeries(LW().LineSeries, opts, o.pane || 0); s.setData(data); return s;
  }
  function addBaseline(chart, data, o = {}) {
    const c = o.color;
    const s = chart.addSeries(LW().BaselineSeries, { baseValue: { type: 'price', price: 0 }, topLineColor: 'rgba(0,0,0,0)', topFillColor1: 'rgba(0,0,0,0)', topFillColor2: 'rgba(0,0,0,0)', bottomLineColor: c, bottomFillColor1: alpha(c, 0.04), bottomFillColor2: alpha(c, 0.35), lineWidth: 1.5, priceLineVisible: false, lastValueVisible: o.last !== false, crosshairMarkerRadius: 3, priceFormat: priceFormat(o) }, o.pane || 0);
    s.setData(data); return s;
  }
  function addStepArea(chart, data, o = {}) {
    const c = o.color;
    const s = chart.addSeries(LW().AreaSeries, { lineColor: c, topColor: alpha(c, 0.38), bottomColor: alpha(c, 0.03), lineWidth: 1.5, lineType: LW().LineType.WithSteps, priceLineVisible: false, lastValueVisible: o.last !== false, crosshairMarkerRadius: 3, priceFormat: priceFormat(o) }, o.pane || 0);
    s.setData(data); return s;
  }
  function addCandles(chart, data, o = {}) {
    const t = theme();
    const s = chart.addSeries(LW().CandlestickSeries, { upColor: t.up, downColor: t.down, borderVisible: false, wickUpColor: t.up, wickDownColor: t.down, priceLineVisible: false, priceFormat: { type: 'price', precision: 2, minMove: 0.01 } }, o.pane || 0);
    s.setData(data);
    onTheme(() => { const t = theme(); s.applyOptions({ upColor: t.up, downColor: t.down, wickUpColor: t.up, wickDownColor: t.down }); });
    return s;
  }
  function addVolume(chart, candles, o = {}) {
    const t = theme();
    const s = chart.addSeries(LW().HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: 'vol', priceLineVisible: false, lastValueVisible: false }, o.pane || 0);
    s.setData(candles.map((c) => ({ time: c.time, value: c.volume, color: alpha(c.close >= c.open ? t.up : t.down, 0.35) })));
    s.priceScale().applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    return s;
  }
  function stretch(chart, factors) {
    try { const panes = chart.panes(); factors.forEach((f, i) => { if (panes[i] && panes[i].setStretchFactor) panes[i].setStretchFactor(f); }); } catch (e) { /* older build */ }
  }
  function bands(chart, container, times) {
    const layer = el('div', 'bands'); container.appendChild(layer);
    let items = [];
    const idxOf = (d) => { const n = times.length - 1; if (n < 0) return 0; if (d <= times[0]) return 0; if (d >= times[n]) return n; let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (times[m] < d) lo = m + 1; else hi = m; } return lo; };
    function draw() {
      layer.innerHTML = '';
      const ts = chart.timeScale();
      let axis = 26, psw = 0;
      try { axis = ts.height(); psw = chart.priceScale('right').width(); } catch (e) { /* defaults */ }
      layer.style.right = psw + 'px'; layer.style.bottom = axis + 'px';
      const inner = container.clientWidth - psw;
      for (const b of items) {
        let x1 = ts.logicalToCoordinate(idxOf(b.from)), x2 = ts.logicalToCoordinate(idxOf(b.to));
        if (x1 == null || x2 == null) continue;
        x1 = Math.max(0, x1); x2 = Math.min(inner, x2);
        if (x2 - x1 < 0.5) continue;
        const d = el('div', 'band ' + b.cls); d.style.left = x1 + 'px'; d.style.width = (x2 - x1) + 'px'; if (b.title) d.title = b.title; layer.appendChild(d);
      }
    }
    chart.timeScale().subscribeVisibleLogicalRangeChange(draw);
    new ResizeObserver(draw).observe(container);
    return { set(list) { items = list || []; draw(); }, draw, layer };
  }
  // A unix time on a chart is Eastern wall-clock read as UTC (see the intraday payload), so
  // formatting it in UTC gives the session clock; a date string is shown as is.
  const clock = (time) => (typeof time === 'number' ? new Date(time * 1000).toLocaleTimeString('en-US', { timeZone: 'UTC', hour: 'numeric', minute: '2-digit' }) + ' ET' : String(time));
  function legend(chart, host, items, times) {
    const box = el('div', 'legend'); host.appendChild(box);
    const dt = el('span', 'dt'); box.appendChild(dt);
    const vals = items.map((it) => { const c = el('span', 'chip'); c.innerHTML = `<span class="sw" style="background:${it.color}"></span>${esc(it.label)} <span class="v"></span>`; box.appendChild(c); return c.querySelector('.v'); });
    function show(param) {
      let hit = false;
      items.forEach((it, i) => { const d = param && param.seriesData ? param.seriesData.get(it.series) : null; if (d) hit = true; const v = d ? (d.value != null ? d.value : d.close) : it.last; vals[i].textContent = it.fmt(v); });
      dt.textContent = hit && param.time ? clock(param.time) : (times && times.length ? clock(times[times.length - 1]) : '');
    }
    chart.subscribeCrosshairMove(show); show(null);
    return box;
  }
  function spans(times, values, pred, cls, title) {
    const out = []; let start = null;
    for (let i = 0; i < times.length; i++) { const on = pred(values[i]); if (on && start == null) start = i; if (!on && start != null) { out.push({ from: times[start], to: times[i], cls, title }); start = null; } }
    if (start != null) out.push({ from: times[start], to: times[times.length - 1], cls, title });
    return out;
  }
  const outBands = (times, exposure) => [...spans(times, exposure, (v) => v != null && v < 0.05, 'out', 'out of the market'), ...spans(times, exposure, (v) => v != null && v >= 0.05 && v < 0.95, 'half', 'partly in')];

  // ---- one run, charted ---------------------------------------------------------
  function runView(root, p, opts = {}) {
    root.innerHTML = '';
    const t = theme(); const S = p.series; const times = S.time; const hasB = !!S.benchmark;
    const m = p.metrics || {}, b = p.benchmark_metrics || {}; const bname = p.benchmark || p.asset;
    const timeIn = S.exposure && S.exposure.length ? S.exposure.reduce((a, v) => a + Math.min(1, Math.abs(v || 0)), 0) / S.exposure.length : null;

    const kp = el('div', 'kpis');
    const sub = (f, k) => (hasB && b[k] != null ? `${esc(bname)} <b>${f(b[k])}</b>` : '');
    const tiles = [
      ['CAGR', fmt.pct(m.cagr), sub(fmt.pct, 'cagr'), m.cagr],
      ['Sharpe', fmt.num(m.sharpe), sub(fmt.num, 'sharpe'), null],
      ['Max drawdown', fmt.pct(m.max_drawdown), sub(fmt.pct, 'max_drawdown'), null],
      ['Volatility', fmt.pct(m.ann_vol), sub(fmt.pct, 'ann_vol'), null],
      ['Total return', fmt.pct(m.total_return, 0), sub((v) => fmt.pct(v, 0), 'total_return'), m.total_return],
      ['Calmar', fmt.num(m.calmar), sub(fmt.num, 'calmar'), null],
      ['Time in market', timeIn == null ? '–' : fmt.pct(timeIn, 0), `<b>${fmt.int(p.n_trades)}</b> trades`, null],
    ];
    for (const [l, v, s, sign] of tiles) { const k = el('div', 'kpi'); k.innerHTML = `<div class="l">${l}</div><div class="v ${signCls(sign)}">${v}</div>${s ? `<div class="s">${s}</div>` : ''}`; kp.appendChild(k); }
    root.appendChild(kp);

    // growth + drawdown + exposure, three panes, one time axis
    const card = el('div', 'chart-card'); const head = el('div', 'chart-head', `<h3>Growth of ${fmt.money(p.initial_capital)}</h3>`); card.appendChild(head);
    const cEl = el('div', 'chart xtall'); card.appendChild(cEl); root.appendChild(card);
    const chart = makeChart(cEl);
    const pts = (arr, scale = 1) => times.map((d, i) => ({ time: d, value: arr[i] == null ? undefined : arr[i] * scale })).filter((x) => x.value !== undefined);
    const eq = addLine(chart, pts(S.equity), { color: t.blue, width: 2, format: fmt.money, minMove: 1 });
    let ghost = null; if (opts.ghost && opts.ghost.time) { ghost = addLine(chart, opts.ghost.time.map((d, i) => ({ time: d, value: opts.ghost.equity[i] })).filter((x) => x.value != null), { color: t.magenta, width: 1, dashed: true, format: fmt.money, minMove: 1, last: false }); }
    let bench = null; if (hasB) bench = addLine(chart, pts(S.benchmark), { color: t.grey, width: 1.5, format: fmt.money, minMove: 1 });
    const dd = addBaseline(chart, pts(S.drawdown, 100), { color: t.down, pane: 1, format: (v) => v.toFixed(0) + '%', minMove: 0.1 });
    if (hasB) addLine(chart, pts(S.benchmark_drawdown, 100), { color: t.grey, width: 1, pane: 1, format: (v) => v.toFixed(0) + '%', minMove: 0.1, last: false });
    const ex = addStepArea(chart, pts(S.exposure, 100), { color: t.aqua, pane: 2, format: (v) => v.toFixed(0) + '%', minMove: 1 });
    stretch(chart, [3.2, 1.1, 0.75]);
    chart.priceScale('right').applyOptions({ mode: LW().PriceScaleMode.Logarithmic });
    const items = [{ series: eq, label: p.strategy, color: t.blue, fmt: fmt.money, last: last(S.equity) }];
    if (ghost) items.push({ series: ghost, label: 'previous run', color: t.magenta, fmt: fmt.money, last: last(opts.ghost.equity) });
    if (bench) items.push({ series: bench, label: `${bname} buy & hold`, color: t.grey, fmt: fmt.money, last: last(S.benchmark) });
    items.push({ series: dd, label: 'drawdown', color: t.down, fmt: (v) => (fin(v) ? v.toFixed(1) + '%' : '–'), last: last(S.drawdown) * 100 });
    items.push({ series: ex, label: 'exposure', color: t.aqua, fmt: (v) => (fin(v) ? v.toFixed(0) + '%' : '–'), last: last(S.exposure) * 100 });
    legend(chart, head, items, times);
    const ctrl = el('div', 'controls'); head.appendChild(ctrl);
    const logBtn = el('button', 'seg-btn', 'Log'); logBtn.type = 'button'; logBtn.setAttribute('aria-pressed', 'true'); ctrl.appendChild(logBtn);
    logBtn.addEventListener('click', () => { const on = logBtn.getAttribute('aria-pressed') !== 'true'; logBtn.setAttribute('aria-pressed', String(on)); chart.priceScale('right').applyOptions({ mode: on ? LW().PriceScaleMode.Logarithmic : LW().PriceScaleMode.Normal }); });
    const bl = bands(chart, cEl, times);
    if (p.windows && p.windows.length) {
      const winBtn = el('button', 'seg-btn', 'Windows'); winBtn.type = 'button'; winBtn.setAttribute('aria-pressed', 'true'); ctrl.appendChild(winBtn);
      const wb = p.windows.filter((w, i) => i % 2 === 0).map((w) => ({ from: w.test_start, to: w.test_end, cls: 'window', title: `test window ${w.window}: ${paramsShort(w.params)}` }));
      bl.set(wb);
      winBtn.addEventListener('click', () => { const on = winBtn.getAttribute('aria-pressed') !== 'true'; winBtn.setAttribute('aria-pressed', String(on)); bl.set(on ? wb : []); });
    }
    chart.timeScale().fitContent();

    // the asset's price with the trades on it
    if (p.candles && p.candles.length) {
      const card2 = el('div', 'chart-card'); const head2 = el('div', 'chart-head', `<h3>${esc(p.asset)} with every trade</h3>`); card2.appendChild(head2);
      const c2 = el('div', 'chart tall'); card2.appendChild(c2); root.appendChild(card2);
      const chart2 = makeChart(c2);
      const cs = addCandles(chart2, p.candles); addVolume(chart2, p.candles);
      const raw = (p.markers || []).filter((x) => x.delta_weight != null); const labelled = raw.length <= 120;
      const mk = raw.map((x) => ({ time: x.time, position: x.delta_weight > 0 ? 'belowBar' : 'aboveBar', color: x.delta_weight > 0 ? t.aqua : t.magenta, shape: x.delta_weight > 0 ? 'arrowUp' : 'arrowDown', size: labelled ? 1 : 0.6, text: labelled ? (x.delta_weight > 0 ? '+' : '') + Math.round(x.delta_weight * 100) + '%' : undefined }));
      if (mk.length) LW().createSeriesMarkers(cs, mk);
      legend(chart2, head2, [{ series: cs, label: p.asset, color: t.grey, fmt: fmt.px, last: last(p.candles.map((c) => c.close)) }], times);
      const ctrl2 = el('div', 'controls'); head2.appendChild(ctrl2);
      const shadeBtn = el('button', 'seg-btn', 'Shade out-of-market'); shadeBtn.type = 'button'; shadeBtn.setAttribute('aria-pressed', 'true'); ctrl2.appendChild(shadeBtn);
      const bl2 = bands(chart2, c2, p.candles.map((c) => c.time)); const ob = outBands(times, S.exposure); bl2.set(ob);
      shadeBtn.addEventListener('click', () => { const on = shadeBtn.getAttribute('aria-pressed') !== 'true'; shadeBtn.setAttribute('aria-pressed', String(on)); bl2.set(on ? ob : []); });
      chart2.timeScale().fitContent();
      chart.timeScale().subscribeVisibleLogicalRangeChange((r) => { if (r) chart2.timeScale().setVisibleLogicalRange(r); });
      chart2.timeScale().subscribeVisibleLogicalRangeChange((r) => { if (r) chart.timeScale().setVisibleLogicalRange(r); });
    }

    // yearly returns + windows
    const two = el('div', 'two'); root.appendChild(two);
    const yt = el('div', 'table');
    yt.innerHTML = `<table class="data"><thead><tr><th>Year</th><th class="num">${esc(p.strategy)}</th>${hasB ? `<th class="num">${esc(bname)}</th><th class="num">Excess</th>` : ''}</tr></thead><tbody>${(p.yearly || []).map((y) => { const ex = hasB && fin(y.benchmark) && fin(y.strategy) ? y.strategy - y.benchmark : null; return `<tr><td class="mono">${y.year}</td><td class="num ${signCls(y.strategy)}">${fmt.pct(y.strategy)}</td>${hasB ? `<td class="num ${signCls(y.benchmark)}">${fmt.pct(y.benchmark)}</td><td class="num ${signCls(ex)}">${fmt.sign(ex)}<span class="bar ${ex < 0 ? 'neg' : ''}" style="width:${Math.min(60, Math.abs(ex || 0) * 150)}px"></span></td>` : ''}</tr>`; }).join('')}</tbody></table>`;
    two.appendChild(yt);
    if (p.windows && p.windows.length) {
      const wt = el('div', 'table');
      wt.innerHTML = `<table class="data"><thead><tr><th>#</th><th>Out-of-sample year</th><th>Chosen settings</th><th class="num">Sharpe</th><th class="num">CAGR</th><th class="num">Max DD</th></tr></thead><tbody>${p.windows.map((w) => `<tr><td class="mono">${w.window}</td><td class="mono small">${w.test_start} → ${w.test_end}</td><td class="small mono">${esc(paramsShort(w.params))}</td><td class="num">${fmt.num(w.oos_sharpe)}</td><td class="num ${signCls(w.oos_cagr)}">${fmt.pct(w.oos_cagr)}</td><td class="num neg">${fmt.pct(w.oos_max_drawdown)}</td></tr>`).join('')}</tbody></table>`;
      two.appendChild(wt);
    } else {
      const info = el('div', 'notice', `<b>How to read this.</b> The blue line is the strategy, grey is buying and holding ${esc(bname)}. The middle pane is how far below its own peak each one sits. The bottom pane is how much of the money was in the market. Costs: ${esc(costsText(p.costs))}; fills at the ${esc(String(p.fill_model || '').replace('_', ' '))}.`);
      two.appendChild(info);
    }
    return { chart };
  }

  function playUrl(p) {
    const q = new URLSearchParams({ strategy: p.strategy, start: p.start, end: p.end, slippage_bps: (p.costs && p.costs.slippage_bps) || 1, fill_model: p.fill_model || 'next_open' });
    const syms = ((p.universe && p.universe.symbols) || []).filter((s) => !s.startsWith('^') && s !== 'CASH');
    if (p.params && p.params.base && p.params.asset) { const lev = String(p.params.asset); const ext = lev.endsWith('.X'); q.set('pair', `${p.params.base}/${ext ? lev.slice(0, -2) : lev}`); q.set('extended', String(ext)); }
    else if (p.params && p.params.asset) q.set('asset', p.params.asset); else if (syms.length) q.set('symbols', syms.join(','));
    for (const [k, v] of Object.entries(p.params || {})) if (!HIDDEN.has(k) && k !== 'asset') q.set('p.' + k, String(v));
    return '/lab/play?' + q.toString();
  }

  // ---- pages --------------------------------------------------------------------
  async function runPage(root) {
    const hash = root.dataset.hash;
    let p;
    try { p = await getJSON(`/lab/api/run/${hash}`); } catch (e) { root.innerHTML = `<h1>Run ${esc(hash)}</h1><div class="notice err">${esc(e.message)}</div>`; return; }
    const study = p.study;
    const uni = (p.universe && p.universe.symbols) || [];
    root.innerHTML = `<h1 class="gradient">${esc(p.strategy)}</h1>
      <div class="meta">${esc(hash)} · ${esc(p.start)} → ${esc(p.end)} · ${esc(uni.length > 8 ? uni.length + ' symbols' : uni.join(', '))} · slippage ${esc((p.costs || {}).slippage_bps)} bps · ${esc(String(p.fill_model || '').replace('_', ' '))} fills${p.git_sha ? ` · ${esc(p.git_sha)}${p.dirty ? '*' : ''}` : ''}${p.created_at ? ` · logged ${esc(String(p.created_at).slice(0, 10))}` : ''}</div>
      <div class="chips" style="margin:6px 0 12px">${study ? `<span class="pill ${study.verdict === 'pass' ? 'pass' : 'fail'}">${esc(study.verdict || 'study')}</span>` : ''}${p.sweep_id ? `<span class="chip">sweep <b>${esc(p.sweep_id)}</b></span>` : ''}${paramChips(p.params)}</div>
      <div class="actions" style="margin:0 0 6px">
        <a class="btn sm" href="${playUrl(p)}">Open in the playground →</a>
        ${study && study.full_hash && study.full_hash !== hash ? `<a class="btn sm quiet" href="/lab/run/${esc(study.full_hash)}">Full-sample run with default settings</a>` : ''}
        <a class="btn sm quiet" href="/lab/chart?symbol=${encodeURIComponent(p.asset)}">${esc(p.asset)} chart</a>
      </div>
      ${study ? `<p class="small muted">Walk-forward: ${study.n_windows} out-of-sample windows, ${study.train_months} months of training for ${study.test_months} months of trading, picking the ${esc(study.objective)}-best settings each time. Shaded stripes mark alternate windows.${study.study ? ` Part of the <b>${esc(study.study)}</b> study${study.pair ? ` (${esc(study.pair)}${study.extended ? ', extended history' : ''})` : ''}.` : ''}</p>` : ''}
      ${writeupCards(p.writeups)}
      <div id="run-view"></div>`;
    runView($('#run-view', root), p);
  }

  // The write-up(s) a run belongs to (ADR-0011): named directly, or about the run's study.
  function writeupCards(list) {
    if (!list || !list.length) return '';
    const ico = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>';
    return list.map((w) => {
      const who = w.ideas && w.ideas.length ? ` Prompted by ${w.ideas.map((i) => `<a href="/ideas#${esc(i.id)}">${esc(i.title)}</a>${i.author ? ' (' + esc(i.author) + ')' : ''}`).join(', ')} in the ideas box.` : '';
      const how = w.how === 'run' ? (w.label ? `This run is <b>${esc(w.label)}</b> in the write-up.` : 'This run is cited in the write-up.') : 'This run is part of the study the write-up reports on.';
      return `<div class="writeup"><div class="ico">${ico}</div><div><div class="meta">Write-up · ${esc(w.date)}</div><h3><a href="/library/${esc(w.slug)}">${esc(w.title)}</a></h3><p>${esc(w.summary || '')}</p><p class="small muted">${how}${who} <a href="/library/${esc(w.slug)}">Read it →</a></p></div></div>`;
    }).join('');
  }

  function browser() {
    const tabs = document.querySelectorAll('.tabs [role=tab]');
    tabs.forEach((b) => b.addEventListener('click', () => { tabs.forEach((x) => x.setAttribute('aria-selected', String(x === b))); $('#tab-studies').hidden = b.dataset.tab !== 'studies'; $('#tab-runs').hidden = b.dataset.tab !== 'runs'; if (b.dataset.tab === 'runs' && !runsLoaded) loadRuns(); }));
    document.querySelectorAll('tr.link[data-href]').forEach((tr) => tr.addEventListener('click', () => { location.href = tr.dataset.href; }));
    // studies: sort by data attributes
    const st = $('#studies');
    st.querySelectorAll('th.sortable').forEach((th) => th.addEventListener('click', () => {
      const k = th.dataset.k; const asc = th.classList.contains('sorted') && !th.classList.contains('asc');
      st.querySelectorAll('th').forEach((x) => x.classList.remove('sorted', 'asc')); th.classList.add('sorted'); if (asc) th.classList.add('asc');
      const rows = [...st.tBodies[0].rows]; rows.sort((a, b) => { const va = a.dataset[k], vb = b.dataset[k]; const na = parseFloat(va), nb = parseFloat(vb); if (isNaN(na) || isNaN(nb)) return (va || '').localeCompare(vb || '') * (asc ? 1 : -1); return (na - nb) * (asc ? 1 : -1); });
      rows.forEach((r) => st.tBodies[0].appendChild(r));
    }));
    // runs: fetched, filtered and sorted client-side
    let runsLoaded = false, runs = [], sortKey = 'created_at', sortAsc = false;
    const tbl = $('#runs'), body = tbl.tBodies[0];
    const val = (r, k) => (k.startsWith('m_') ? (r.metrics || {})[k.slice(2)] : r[k]);
    function render() {
      const q = ($('#run-search').value || '').toLowerCase();
      let rows = runs.filter((r) => !q || `${r.hash} ${r.strategy} ${paramsShort(r.params)} ${r.universe}`.toLowerCase().includes(q));
      rows.sort((a, b) => { const va = val(a, sortKey), vb = val(b, sortKey); if (va == null && vb == null) return 0; if (va == null) return 1; if (vb == null) return -1; const c = typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb)); return sortAsc ? c : -c; });
      $('#run-count').textContent = `${rows.length} of ${runs.length} runs`;
      body.innerHTML = rows.map((r) => `<tr class="link" data-href="/lab/run/${r.hash}"><td class="mono small">${r.hash}</td><td><b>${esc(r.strategy)}</b></td><td class="small mono">${esc(paramsShort(r.params))}</td><td class="small mono">${esc(r.universe.length > 28 ? r.universe.slice(0, 26) + '…' : r.universe)}</td><td class="small mono">${r.start} → ${r.end}</td><td class="num">${fmt.num(r.metrics.sharpe)}</td><td class="num ${signCls(r.metrics.cagr)}">${fmt.pct(r.metrics.cagr)}</td><td class="num neg">${fmt.pct(r.metrics.max_drawdown)}</td><td class="num">${fmt.int(r.metrics.n_trades)}</td><td class="small mono">${String(r.created_at || '').slice(0, 10)}</td></tr>`).join('') || '<tr><td colspan="10" class="muted">No runs match.</td></tr>';
      body.querySelectorAll('tr.link').forEach((tr) => tr.addEventListener('click', () => { location.href = tr.dataset.href; }));
    }
    async function loadRuns() {
      runsLoaded = true; body.innerHTML = '<tr><td colspan="10"><span class="spin"></span></td></tr>';
      const q = new URLSearchParams({ standalone: String($('#run-standalone').checked), limit: '500' });
      const s = $('#run-strategy').value; if (s) q.set('strategy', s);
      try { runs = (await getJSON('/lab/api/runs?' + q)).runs; } catch (e) { body.innerHTML = `<tr><td colspan="10" class="neg">${esc(e.message)}</td></tr>`; return; }
      render();
    }
    tbl.querySelectorAll('th.sortable').forEach((th) => th.addEventListener('click', () => { const k = th.dataset.k; sortAsc = sortKey === k ? !sortAsc : false; sortKey = k; tbl.querySelectorAll('th').forEach((x) => x.classList.remove('sorted', 'asc')); th.classList.add('sorted'); if (sortAsc) th.classList.add('asc'); render(); }));
    $('#run-strategy').addEventListener('change', loadRuns); $('#run-standalone').addEventListener('change', loadRuns);
    $('#run-search').addEventListener('input', debounce(render, 150));
    if (location.hash === '#runs') tabs[1].click();
  }

  // ---- playground ---------------------------------------------------------------
  function playground(root, catalog, presets, pairs) {
    pairs = pairs || [];
    const byKey = Object.fromEntries(catalog.map((s) => [s.key, s]));
    const f = { strategy: $('#f-strategy'), doc: $('#f-doc'), uni: $('#f-universe'), start: $('#f-start'), end: $('#f-end'), slip: $('#f-slip'), fill: $('#f-fill'), params: $('#f-params'), run: $('#f-run'), reset: $('#f-reset'), auto: $('#f-auto'), log: $('#f-log'), hist: $('#f-history'), form: $('#play-form') };
    const main = $('#play-main');
    const results = new Map(); const order = []; let current = null, lastPayload = null, inflight = 0;
    const q = new URLSearchParams(location.search);

    const KIND_LABEL = { dial: 'In or out of one asset (steered by the VIX family)', leverage: 'Leveraged fund when the market turns bullish, plain fund otherwise', portfolio: 'Portfolio strategies (a basket of symbols)' };
    f.strategy.innerHTML = ['dial', 'leverage', 'portfolio'].map((kind) => `<optgroup label="${KIND_LABEL[kind]}">${catalog.filter((s) => s.kind === kind).map((s) => `<option value="${s.key}">${s.key}</option>`).join('')}</optgroup>`).join('');
    const initial = q.get('strategy') && byKey[q.get('strategy')] ? q.get('strategy') : (byKey.vix_vote ? 'vix_vote' : catalog[0].key);
    f.strategy.value = initial;
    if (q.get('start')) f.start.value = q.get('start'); if (q.get('end')) f.end.value = q.get('end');
    if (q.get('slippage_bps')) f.slip.value = q.get('slippage_bps'); if (q.get('fill_model')) f.fill.value = q.get('fill_model');

    function buildUniverse(s) {
      if (s.kind === 'leverage') {
        const cur = (q.get('pair') || 'QQQ/TQQQ').toUpperCase(); const ext = q.get('extended') !== 'false';
        f.uni.innerHTML = `<div class="field"><label>Pair (plain fund / leveraged fund)</label><select id="f-pair">${pairs.map((p) => `<option value="${esc(p.key)}"${p.key === cur ? ' selected' : ''}>${esc(p.key)} — ${esc(p.name)}</option>`).join('')}</select></div><label class="check" style="margin:-6px 0 10px"><input type="checkbox" id="f-extended" ${ext ? 'checked' : ''}> Extended history (synthetic rebuild spliced before the fund existed)</label><div class="small muted" style="margin:-4px 0 12px">The rule reads the plain fund; the money sits in the leveraged fund when it says "bull", in the plain fund otherwise.</div>`;
        $('#f-pair').addEventListener('change', schedule); $('#f-extended').addEventListener('change', schedule);
      } else if (s.kind === 'dial') {
        f.uni.innerHTML = `<div class="field"><label>Asset (any ticker)</label><input type="text" id="f-asset" value="${esc(q.get('asset') || 'SPY')}" style="font-family:var(--mono);font-weight:600;text-transform:uppercase"></div><div class="small muted" style="margin:-6px 0 12px">Held against cash earning the T-bill rate. The VIX family rides along as the signal.</div>`;
      } else {
        const preset = presets.find((p) => p.name === 'etf_core') || presets[0];
        const syms = q.get('symbols') || (preset ? preset.symbols.join(', ') : 'SPY, QQQ, IWM, TLT, GLD');
        f.uni.innerHTML = `<div class="field"><label>Symbols</label><textarea id="f-symbols" rows="3">${esc(syms)}</textarea></div><div class="field"><label>Preset</label><select id="f-preset"><option value="">— pick a basket —</option>${presets.map((p) => `<option value="${esc(p.name)}">${esc(p.name)} (${p.symbols.length})</option>`).join('')}</select></div>`;
        $('#f-preset').addEventListener('change', (e) => { const p = presets.find((x) => x.name === e.target.value); if (p) { $('#f-symbols').value = p.symbols.join(', '); schedule(); } });
      }
    }
    function buildParams(s, values) {
      f.params.innerHTML = '';
      if (!s.params.length) { f.params.innerHTML = '<div class="small muted">No knobs on this one.</div>'; return; }
      for (const p of s.params) {
        const v = values && values[p.name] != null ? values[p.name] : p.default;
        if (p.type === 'bool') { const w = el('label', 'check'); w.innerHTML = `<input type="checkbox" data-p="${p.name}" ${v ? 'checked' : ''}> ${esc(p.name)}`; w.style.margin = '0 0 10px'; f.params.appendChild(w); continue; }
        if (p.type === 'str') { const w = el('div', 'field'); w.innerHTML = `<label>${esc(p.name)}</label><input type="text" data-p="${p.name}" value="${esc(v)}">`; f.params.appendChild(w); continue; }
        const w = el('div', 'slider');
        const step = p.step, min = p.min, max = p.max;
        // The typed number is the value that runs; the slider is a convenience that snaps to
        // ~40 stops. The number box itself takes any value (step "any" for floats), inside
        // or outside the slider's range, so the browser never rejects a hand-typed 30.
        w.innerHTML = `<div class="lbl"><span>${esc(p.name)}</span><span class="d">default ${esc(p.default)}</span></div><input type="range" data-r="${p.name}" min="${min}" max="${max}" step="${step}" value="${v}"><input type="number" data-p="${p.name}" data-default="${esc(p.default)}" step="${p.type === 'int' ? 1 : 'any'}" value="${v}">${p.grid ? `<div class="grid-marks">study grid: ${p.grid.join(' · ')}</div>` : ''}`;
        const r = w.querySelector('[type=range]'), n = w.querySelector('[type=number]');
        r.addEventListener('input', () => { n.value = r.value; scheduleSlider(); });
        n.addEventListener('input', () => { if (n.value !== '' && isFinite(Number(n.value))) { r.value = n.value; scheduleSlider(); } });
        f.params.appendChild(w);
      }
    }
    function readParams() {
      const out = {};
      f.params.querySelectorAll('[data-p]').forEach((i) => {
        const k = i.dataset.p;
        if (i.type === 'checkbox') out[k] = i.checked;
        else if (i.type === 'number') { const n = Number(i.value); out[k] = i.value !== '' && isFinite(n) ? n : Number(i.dataset.default); }
        else out[k] = i.value;
      });
      return out;
    }
    function request() {
      const s = byKey[f.strategy.value];
      const body = { strategy: s.key, params: readParams(), start: f.start.value, end: f.end.value, slippage_bps: Number(f.slip.value), fill_model: f.fill.value, log: !!(f.log && f.log.checked) };
      if (s.kind === 'leverage') { body.pair = $('#f-pair').value; body.extended = $('#f-extended').checked; }
      else if (s.kind === 'dial') body.asset = ($('#f-asset').value || 'SPY').trim().toUpperCase(); else body.symbols = ($('#f-symbols').value || '').split(/[\s,]+/).filter(Boolean).map((x) => x.toUpperCase());
      return body;
    }
    function syncUrl(body) {
      const u = new URLSearchParams({ strategy: body.strategy, start: body.start, end: body.end, slippage_bps: String(body.slippage_bps), fill_model: body.fill_model });
      if (body.asset) u.set('asset', body.asset); if (body.symbols) u.set('symbols', body.symbols.join(',')); if (body.pair) { u.set('pair', body.pair); u.set('extended', String(body.extended)); }
      for (const [k, v] of Object.entries(body.params)) u.set('p.' + k, String(v));
      history.replaceState(null, '', '/lab/play?' + u.toString());
    }
    function renderHistory() {
      if (!order.length) return;
      f.hist.innerHTML = '';
      for (const h of [...order].reverse()) {
        const p = results.get(h); const b = el('button'); b.type = 'button'; b.setAttribute('aria-current', String(h === current));
        b.innerHTML = `<span>${esc(p.strategy)}${p.params.asset ? ' · ' + esc(p.params.asset) : ''}${p.params.base ? '/' + esc(p.params.base) : ''}<br><span class="faint">${esc(paramsShort(p.params).slice(0, 48))}</span></span><span class="m">${fmt.num(p.metrics.sharpe)}<br><span class="${signCls(p.metrics.cagr)}">${fmt.pct(p.metrics.cagr)}</span></span>`;
        b.addEventListener('click', () => show(h, false));
        f.hist.appendChild(b);
      }
    }
    function show(h, ghost) {
      const p = results.get(h); if (!p) return;
      const prev = ghost && lastPayload && lastPayload.hash !== h ? lastPayload.series : null;
      current = h; main.innerHTML = '';
      const hdr = el('div', '', `<div class="meta">${esc(p.hash)} · ${esc(p.start)} → ${esc(p.end)} ${p.logged ? '· <span class="pill pass">logged</span>' : '· <span class="pill neutral">not saved</span>'}</div><div class="chips" style="margin:4px 0 10px">${paramChips(p.params)}</div>`);
      main.appendChild(hdr);
      const view = el('div'); main.appendChild(view);
      runView(view, p, { ghost: prev });
      lastPayload = p; renderHistory();
    }
    async function run() {
      let body; try { body = request(); } catch (e) { return; }
      syncUrl(body);
      f.run.disabled = true; f.run.innerHTML = '<span class="spin"></span> Running'; inflight++;
      const stamp = ++runSeq;
      try {
        const p = await getJSON('/lab/api/play', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
        if (stamp !== runSeq) return; // a newer run superseded this one
        if (!results.has(p.hash)) order.push(p.hash); results.set(p.hash, p);
        show(p.hash, true);
      } catch (e) {
        if (stamp === runSeq) { const n = el('div', 'notice err', `<b>That run failed.</b> ${esc(e.message)}`); main.prepend(n); }
      } finally { if (--inflight === 0) { f.run.disabled = false; f.run.textContent = 'Run'; } }
    }
    let runSeq = 0;
    const schedule = debounce(() => { if (f.auto.checked) run(); }, 250);
    const scheduleSlider = debounce(() => { if (f.auto.checked) run(); }, 650);
    function rebuild(values) { const s = byKey[f.strategy.value]; f.doc.textContent = s.doc; buildUniverse(s); buildParams(s, values); $('#f-asset, #f-symbols') && $('#f-asset, #f-symbols').addEventListener('change', schedule); }
    const fromQuery = {}; for (const [k, v] of q.entries()) if (k.startsWith('p.')) fromQuery[k.slice(2)] = v === 'true' ? true : v === 'false' ? false : v;
    rebuild(fromQuery);
    f.strategy.addEventListener('change', () => { rebuild(null); schedule(); });
    [f.start, f.end, f.slip, f.fill].forEach((i) => i.addEventListener('change', schedule));
    f.reset.addEventListener('click', () => { rebuild(null); schedule(); });
    f.form.addEventListener('submit', (e) => { e.preventDefault(); run(); });
    if (q.has('strategy') || q.has('asset') || q.has('pair')) run();
  }

  // ---- any symbol ---------------------------------------------------------------
  // --- freshness: when the data was pulled, and how far behind the tape it is ---------
  const ET = 'America/New_York';
  function etTime(iso) { if (!iso) return '–'; return new Date(iso).toLocaleTimeString('en-US', { timeZone: ET, hour: 'numeric', minute: '2-digit' }); }
  function etDate(iso) { if (!iso) return ''; return new Date(iso).toLocaleDateString('en-US', { timeZone: ET, month: 'short', day: 'numeric' }); }
  function marketOpen(now) {
    // US cash session, Mon–Fri 9:30–16:00 ET (holidays are treated as open days; the
    // only cost is a "provisional" label on a bar that is in fact final).
    const parts = new Intl.DateTimeFormat('en-US', { timeZone: ET, weekday: 'short', hour: 'numeric', minute: 'numeric', hour12: false }).formatToParts(now);
    const g = (k) => parts.find((x) => x.type === k).value;
    const wd = g('weekday'), m = Number(g('hour')) * 60 + Number(g('minute'));
    return !['Sat', 'Sun'].includes(wd) && m >= 9 * 60 + 30 && m < 16 * 60;
  }
  function freshness(live, lastBarDate, now = new Date(), intraday = false) {
    const open = marketOpen(now);
    const q = live && live.quote;
    const bits = [];
    if (q && q.time) bits.push(`live quote <b>${fmt.px(q.price)}</b> ${q.change_pct != null ? `<span class="${signCls(q.change_pct)}">${fmt.sign(q.change_pct, 2)}</span>` : ''} at ${etTime(q.time)} ET`);
    if (!intraday && live && live.bars_pulled_at) bits.push(`bars pulled ${etTime(live.bars_pulled_at)} ET`);
    let lag;
    if (intraday) lag = open ? 'minute bars run about a minute behind the tape, the quote a few seconds' : 'the market is closed, so these bars are final';
    else lag = open ? 'during the session the latest daily bar is provisional (vendor end-of-day history runs a few minutes behind; the quote a few seconds)' : `daily bars are final through ${esc(lastBarDate || '')}`;
    return `${bits.join(' · ')}${bits.length ? ' · ' : ''}${lag}.`;
  }

  function symbolPage(root) {
    const symbol = root.dataset.symbol, today = root.dataset.today;
    const q = new URLSearchParams(location.search);
    let years = Number(q.get('years') || 5);
    let mode = q.get('range') === 'today' ? 'today' : 'daily';
    let timer = null; // auto-refresh while the Today tab is up during the session
    // Indicator settings live in the URL (so a link carries them) and in the browser's
    // memory (so the next symbol opens with the same lines). Empty string = off.
    const IND_KEYS = ['sma', 'ema', 'bb', 'rsi'];
    let remembered = {}; try { remembered = JSON.parse(localStorage.getItem('ehf.chart.ind') || '{}'); } catch (e) { /* no storage */ }
    const ind = { sma: '50, 200', ema: '', bb: '', rsi: '' };
    for (const k of IND_KEYS) { if (q.has(k)) ind[k] = q.get(k); else if (remembered[k] != null) ind[k] = remembered[k]; }
    function indQuery() { return IND_KEYS.map((k) => `&${k}=${encodeURIComponent(ind[k].trim())}`).join(''); }
    function rememberInd() { for (const k of IND_KEYS) q.set(k, ind[k].trim()); history.replaceState(null, '', '/lab/chart?' + q.toString()); try { localStorage.setItem('ehf.chart.ind', JSON.stringify(ind)); } catch (e) { /* no storage */ } }
    const reloadIndicators = debounce(() => { rememberInd(); load(); }, 450);
    function indicatorBar() {
      const bar = el('div', 'ind-bar');
      const field = (k, label, placeholder, width) => `<label class="ind"><span>${label}</span><input type="text" data-ind="${k}" value="${esc(ind[k])}" placeholder="${placeholder}" style="width:${width}px" spellcheck="false"></label>`;
      bar.innerHTML = `<span class="ind-title">Indicators</span>${field('sma', 'SMA', 'e.g. 50, 200', 92)}${field('ema', 'EMA', 'e.g. 21', 72)}${field('bb', 'Bollinger', 'e.g. 20, 2', 96)}${field('rsi', 'RSI', 'e.g. 14', 56)}<span class="small muted">days; blank turns a line off; up to four per kind</span>`;
      bar.querySelectorAll('[data-ind]').forEach((i) => i.addEventListener('input', () => { ind[i.dataset.ind] = i.value; reloadIndicators(); }));
      return bar;
    }
    const refreshBtn = $('#sym-refresh');
    if (refreshBtn) refreshBtn.addEventListener('click', () => (mode === 'today' ? loadToday(true) : load(true)));
    const rangeBtns = document.querySelectorAll('#sym-range button');
    function select(b) { rangeBtns.forEach((x) => x.setAttribute('aria-selected', String(x === b))); }
    rangeBtns.forEach((b) => {
      b.setAttribute('aria-selected', String(b.dataset.mode ? mode === 'today' : mode === 'daily' && Number(b.dataset.years) === years));
      b.addEventListener('click', () => {
        select(b);
        if (b.dataset.mode === 'today') { mode = 'today'; q.set('range', 'today'); history.replaceState(null, '', '/lab/chart?' + q.toString()); loadToday(); return; }
        mode = 'daily'; years = Number(b.dataset.years); q.delete('range'); q.set('years', String(years)); history.replaceState(null, '', '/lab/chart?' + q.toString()); load();
      });
    });
    $('#sym-form').addEventListener('submit', (e) => { e.preventDefault(); const s = $('#sym-input').value.trim().toUpperCase(); if (s) location.href = `/lab/chart?symbol=${encodeURIComponent(s)}&years=${years}${mode === 'today' ? '&range=today' : ''}`; });
    function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }
    async function loadToday(refresh = false) {
      stopTimer();
      root.innerHTML = `<h1 class="gradient">${esc(symbol)}</h1><div class="notice"><span class="spin"></span> &nbsp;${refresh ? 'Pulling' : 'Loading'} today's minute bars for ${esc(symbol)}.</div>`;
      if (refreshBtn) refreshBtn.disabled = true;
      let d;
      try { d = await getJSON(`/lab/api/intraday/${encodeURIComponent(symbol)}${refresh ? '?refresh=1' : ''}`); } catch (e) { root.innerHTML = `<h1>${esc(symbol)}</h1><div class="notice err">${esc(e.message)}</div>`; if (refreshBtn) refreshBtn.disabled = false; return; }
      if (refreshBtn) refreshBtn.disabled = false;
      if (mode !== 'today') return; // the user moved on while we were fetching
      const t = theme(); const S = d.stats || {}; const open = marketOpen(new Date());
      const sessionIsToday = d.session === today;
      root.innerHTML = `<h1 class="gradient">${esc(symbol)}</h1>
        <div class="kpis">
          <div class="kpi"><div class="l">Last</div><div class="v">${fmt.px(S.last)}</div><div class="s">${esc(S.last_time ? S.last_time.slice(11, 16) : '')} ET${sessionIsToday ? '' : ' · ' + esc(d.session)}</div></div>
          <div class="kpi"><div class="l">vs prior close</div><div class="v ${signCls(S.change)}">${fmt.sign(S.change, 2)}</div><div class="s">from ${fmt.px(d.prev_close)}</div></div>
          <div class="kpi"><div class="l">Session range</div><div class="v" style="font-size:17px">${fmt.px(S.low)} – ${fmt.px(S.high)}</div><div class="s">opened ${fmt.px(S.open)}</div></div>
          <div class="kpi"><div class="l">Volume</div><div class="v">${fmt.vol ? fmt.vol(S.volume) : Math.round(S.volume).toLocaleString()}</div><div class="s">${S.bars} one-minute bars</div></div>
          ${d.vix.length ? `<div class="kpi"><div class="l">VIX now</div><div class="v">${fmt.num(d.vix[d.vix.length - 1].value, 2)}</div><div class="s">implied daily move ≈ ${fmt.num(d.vix[d.vix.length - 1].value / 16, 2)}%</div></div>` : ''}
        </div>
        <p class="small muted" id="sym-fresh">${freshness(d.live, d.session, new Date(), true)} ${open && sessionIsToday ? 'This view re-pulls itself every minute while the market is open.' : sessionIsToday ? 'The session is over; this is the full day.' : `Showing the last session, ${esc(d.session)}; the market is closed.`}</p>
        <div class="chart-card"><div class="chart-head" id="sym-head"><h3>${esc(symbol)}, ${esc(d.session)}, one-minute bars</h3></div><div class="chart xtall" id="sym-chart"></div></div>
        <p class="small muted">Regular session only (9:30 to 4:00 ET). Each candle is one minute, about a minute behind the tape; the dashed line is the prior close. ${d.vix.length ? 'The lower pane is the VIX over the same minutes.' : ''}</p>`;
      const host = $('#sym-chart'), head = $('#sym-head');
      const chart = makeChart(host);
      chart.applyOptions({ timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 5 } });
      const cs = addCandles(chart, d.candles); addVolume(chart, d.candles);
      if (d.prev_close != null) cs.createPriceLine({ price: d.prev_close, color: t.grey, lineWidth: 1, lineStyle: LW().LineStyle.Dashed, axisLabelVisible: true, title: 'prior close' });
      const items = [{ series: cs, label: 'close', color: t.grey, fmt: fmt.px, last: S.last }];
      if (d.vix.length) { const v = addLine(chart, d.vix, { color: t.yellow, width: 1.5, pane: 1, format: (x) => x.toFixed(2), minMove: 0.01 }); items.push({ series: v, label: 'VIX', color: t.yellow, fmt: (x) => fmt.num(x, 2), last: last(d.vix.map((x) => x.value)) }); stretch(chart, [3, 1]); }
      legend(chart, head, items, d.candles.map((c) => c.time));
      chart.timeScale().fitContent();
      if (open && sessionIsToday) timer = setInterval(() => { if (document.visibilityState === 'visible' && mode === 'today') loadToday(); }, 60000);
    }
    async function load(refresh = false) {
      stopTimer();
      const start = years ? new Date(Date.parse(today) - years * 365.25 * 864e5).toISOString().slice(0, 10) : '2000-01-03';
      root.innerHTML = `<h1 class="gradient">${esc(symbol)}</h1><div class="notice"><span class="spin"></span> &nbsp;${refresh ? 'Pulling the latest bars for' : 'Loading'} ${esc(symbol)} since ${start}.</div>`;
      if (refreshBtn) refreshBtn.disabled = true;
      let d;
      try { d = await getJSON(`/lab/api/bars/${encodeURIComponent(symbol)}?start=${start}&end=${today}${refresh ? '&refresh=1' : ''}${indQuery()}`); } catch (e) { root.innerHTML = `<h1>${esc(symbol)}</h1><div class="notice err">${esc(e.message)}</div>`; root.appendChild(indicatorBar()); if (refreshBtn) refreshBtn.disabled = false; return; }
      if (refreshBtn) refreshBtn.disabled = false;
      if (mode !== 'daily') return;
      const t = theme(); const L = d.last || {};
      const O = d.overlays || {}; const I = d.indicators || { sma: [], ema: [], bb: null, rsi: null };
      const palette = [t.orange, t.blue, t.magenta, t.aqua];
      const timeIn = d.exposure.length ? d.exposure.reduce((a, x) => a + x.value, 0) / d.exposure.length : null;
      const cur = d.exposure.length ? d.exposure[d.exposure.length - 1].value : null;
      root.innerHTML = `<h1 class="gradient">${esc(symbol)}</h1>
        <div class="kpis">
          <div class="kpi"><div class="l">Last close</div><div class="v">${fmt.px(L.close)}</div><div class="s">${esc(L.date || '')}</div></div>
          <div class="kpi"><div class="l">Day</div><div class="v ${signCls(L.change)}">${fmt.sign(L.change, 2)}</div></div>
          <div class="kpi"><div class="l">52-week range</div><div class="v" style="font-size:17px">${fmt.px(L.low_52w)} – ${fmt.px(L.high_52w)}</div></div>
          ${d.vix.length ? `<div class="kpi"><div class="l">VIX</div><div class="v">${fmt.num(d.vix[d.vix.length - 1].value, 1)}</div><div class="s">implied daily move ≈ ${fmt.num(d.vix[d.vix.length - 1].value / 16, 2)}%</div></div>` : ''}
          ${cur != null ? `<div class="kpi"><div class="l">Fear-gauge dial today</div><div class="v ${cur >= 0.95 ? 'pos' : cur < 0.05 ? 'neg' : ''}">${Math.round(cur * 100)}% in</div><div class="s">${esc(d.dial)} · in the market ${fmt.pct(timeIn, 0)} of this span</div></div>` : ''}
        </div>
        <p class="small muted" id="sym-fresh">${freshness(d.live, L.date)}</p>
        <div id="sym-ind"></div>
        <div class="chart-card"><div class="chart-head" id="sym-head"><h3>${esc(symbol)}, daily</h3></div><div class="chart xtall" id="sym-chart"></div></div>
        <p class="small muted">Candles are adjusted daily bars. Solid lines are simple moving averages, dashed ones exponential; Bollinger bands are the grey envelope${I.bb ? ` (${I.bb.window}-day, ${I.bb.n_std}σ)` : ''}. ${I.rsi ? `RSI (${I.rsi.window}) has its own pane with the 30 and 70 lines. ` : ''}${d.vix.length ? 'The VIX pane is the fear gauge; the bottom pane is how much of the money the <b>' + esc(d.dial) + '</b> dial would have kept in ' + esc(symbol) + ' on each day, with the out-of-market days shaded on the price.' : 'The VIX family was not available for this symbol, so no dial is drawn.'}</p>`;
      $('#sym-ind').appendChild(indicatorBar());
      const host = $('#sym-chart'), head = $('#sym-head');
      const chart = makeChart(host);
      const cs = addCandles(chart, d.candles); addVolume(chart, d.candles);
      const items = [{ series: cs, label: 'close', color: t.grey, fmt: fmt.px, last: L.close }];
      const pxFmt = { type: 'price', precision: 2, minMove: 0.01 };
      I.sma.forEach((w, i) => { const data = O[`sma${w}`] || []; const ln = addLine(chart, data, { color: palette[i % palette.length], width: 1.5, last: false, priceFormat: pxFmt }); items.push({ series: ln, label: `SMA ${w}`, color: palette[i % palette.length], fmt: fmt.px, last: last(data.map((x) => x.value)) }); });
      I.ema.forEach((w, i) => { const data = O[`ema${w}`] || []; const c = palette[(i + I.sma.length) % palette.length]; const ln = addLine(chart, data, { color: c, width: 1.5, last: false, dashed: true, priceFormat: pxFmt }); items.push({ series: ln, label: `EMA ${w}`, color: c, fmt: fmt.px, last: last(data.map((x) => x.value)) }); });
      if (O.bb) {
        const up = addLine(chart, O.bb.upper, { color: alpha(t.grey, 0.8), width: 1, last: false, priceFormat: pxFmt });
        addLine(chart, O.bb.mid, { color: alpha(t.grey, 0.5), width: 1, last: false, dashed: true, priceFormat: pxFmt });
        const lo = addLine(chart, O.bb.lower, { color: alpha(t.grey, 0.8), width: 1, last: false, priceFormat: pxFmt });
        items.push({ series: up, label: 'BB upper', color: t.grey, fmt: fmt.px, last: last(O.bb.upper.map((x) => x.value)) }, { series: lo, label: 'BB lower', color: t.grey, fmt: fmt.px, last: last(O.bb.lower.map((x) => x.value)) });
      }
      let pane = 1;
      const factors = [3.4];
      if (O.rsi) {
        const r = addLine(chart, O.rsi.values, { color: t.magenta, width: 1.5, pane, format: (x) => x.toFixed(0), minMove: 1 });
        for (const lvl of [30, 70]) r.createPriceLine({ price: lvl, color: alpha(t.grey, 0.6), lineWidth: 1, lineStyle: LW().LineStyle.Dashed, axisLabelVisible: false });
        items.push({ series: r, label: `RSI ${O.rsi.window}`, color: t.magenta, fmt: (x) => (fin(x) ? x.toFixed(0) : '–'), last: last(O.rsi.values.map((x) => x.value)) });
        pane++; factors.push(0.9);
      }
      if (d.vix.length) { const v = addLine(chart, d.vix, { color: t.yellow, width: 1.5, pane, format: (x) => x.toFixed(1), minMove: 0.1 }); items.push({ series: v, label: 'VIX', color: t.yellow, fmt: (x) => fmt.num(x, 1), last: last(d.vix.map((x) => x.value)) }); pane++; factors.push(1); }
      if (d.exposure.length) { const e = addStepArea(chart, d.exposure.map((x) => ({ time: x.time, value: x.value * 100 })), { color: t.aqua, pane, format: (x) => x.toFixed(0) + '%', minMove: 1 }); items.push({ series: e, label: 'dial', color: t.aqua, fmt: (x) => (fin(x) ? x.toFixed(0) + '%' : '–'), last: cur * 100 }); pane++; factors.push(0.7); }
      stretch(chart, factors);
      legend(chart, head, items, d.candles.map((c) => c.time));
      const ctrl = el('div', 'controls'); head.appendChild(ctrl);
      const bl = bands(chart, host, d.candles.map((c) => c.time));
      if (d.exposure.length) {
        const ob = outBands(d.exposure.map((x) => x.time), d.exposure.map((x) => x.value)); bl.set(ob);
        const sb = el('button', 'seg-btn', 'Shade out-of-market'); sb.type = 'button'; sb.setAttribute('aria-pressed', 'true'); ctrl.appendChild(sb);
        sb.addEventListener('click', () => { const on = sb.getAttribute('aria-pressed') !== 'true'; sb.setAttribute('aria-pressed', String(on)); bl.set(on ? ob : []); });
      }
      const lb = el('button', 'seg-btn', 'Log'); lb.type = 'button'; lb.setAttribute('aria-pressed', 'false'); ctrl.appendChild(lb);
      lb.addEventListener('click', () => { const on = lb.getAttribute('aria-pressed') !== 'true'; lb.setAttribute('aria-pressed', String(on)); chart.priceScale('right').applyOptions({ mode: on ? LW().PriceScaleMode.Logarithmic : LW().PriceScaleMode.Normal }); });
      chart.timeScale().fitContent();
    }
    if (mode === 'today') loadToday(); else load();
  }

  // ---- the playbook: one chart per rule ---------------------------------------
  function playbookPage(p) {
    const fresh = $('#pb-fresh');
    if (fresh) { const lq = p.live && p.live[p.asset]; fresh.innerHTML = freshness({ quote: lq, bars_pulled_at: p.built_at }, p.as_of); }
    const t = theme(); const times = p.series.time;
    const price = times.map((d, i) => ({ time: d, value: p.series.close[i] })).filter((x) => x.value != null);
    const charts = [];
    for (const r of p.rules) {
      const host = $('#chart-' + r.key), head = $('#head-' + r.key); if (!host) continue;
      const chart = makeChart(host);
      const px = addLine(chart, price, { color: t.grey, width: 1.5, priceFormat: { type: 'price', precision: 2, minMove: 0.01 } });
      const isPct = r.reading_fmt === 'pct';
      const rf = isPct ? (v) => (fin(v) ? (v * 100).toFixed(0) + '%' : '–') : (v) => (fin(v) ? v.toFixed(3) : '–');
      const scale = isPct ? 100 : 1;
      const reading = r.series.time.map((d, i) => ({ time: d, value: r.series.reading[i] == null ? undefined : r.series.reading[i] * scale })).filter((x) => x.value !== undefined);
      const rd = addLine(chart, reading, { color: t.blue, width: 2, pane: 1, format: isPct ? (v) => v.toFixed(0) + '%' : (v) => v.toFixed(2), minMove: isPct ? 1 : 0.01 });
      const flat = (v) => r.series.time.map((d) => ({ time: d, value: v * scale }));
      addLine(chart, flat(r.out_above), { color: t.down, width: 1, dashed: true, pane: 1, last: false, format: isPct ? (v) => v.toFixed(0) + '%' : (v) => v.toFixed(2), minMove: isPct ? 1 : 0.01 });
      addLine(chart, flat(r.in_below), { color: t.up, width: 1, dashed: true, pane: 1, last: false, format: isPct ? (v) => v.toFixed(0) + '%' : (v) => v.toFixed(2), minMove: isPct ? 1 : 0.01 });
      stretch(chart, [2.2, 1.3]);
      legend(chart, head, [
        { series: px, label: p.asset, color: t.grey, fmt: fmt.px, last: last(p.series.close) },
        { series: rd, label: r.title.toLowerCase(), color: t.blue, fmt: (v) => rf(fin(v) ? v / scale : v), last: r.reading == null ? null : r.reading * scale },
      ], times);
      const bl = bands(chart, host, times); bl.set(outBands(r.series.time, r.series.exposure));
      chart.timeScale().fitContent();
      charts.push(chart);
    }
    charts.forEach((a) => a.timeScale().subscribeVisibleLogicalRangeChange((rg) => { if (rg) charts.forEach((b) => { if (b !== a) b.timeScale().setVisibleLogicalRange(rg); }); }));
  }

  return { runPage, browser, playground, symbolPage, runView, playbookPage, freshness, fmt };
})();
