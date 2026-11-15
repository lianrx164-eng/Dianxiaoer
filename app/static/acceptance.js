/* 后台页共用小工具:DOM 构造、JSON 请求、状态药丸。 */
window.AC = (function () {
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (k === 'html') node.innerHTML = v;
        else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else if (v !== undefined && v !== null) node.setAttribute(k, v);
      }
    }
    (children || []).forEach((c) => {
      if (c === null || c === undefined || c === '') return;   // 条件子节点常写 `cond ? el() : ''`
      node.appendChild(c);
    });
    return node;
  }

  async function getJSON(url, options) {
    const resp = await fetch(url, options);
    let body = null;
    try { body = await resp.json(); } catch (_) { /* 非 JSON */ }
    if (!resp.ok) {
      const detail = body && body.detail ? body.detail : resp.status + ' ' + resp.statusText;
      throw new Error(detail);
    }
    return body;
  }

  function pill(text, kind) {
    return el('span', { class: 'pill ' + (kind || 'ok'), text: text });
  }

  function fmt(n) { return n === null || n === undefined ? '—' : String(n); }

  function fmtBytes(n) {
    if (n === null || n === undefined) return '—';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(1) + ' MB';
  }

  function statusPill(status) {
    const map = { pass: ['通过', 'ok'], fail: ['未过线', 'bad'], missing: ['还没跑过', 'warn'] };
    const [text, kind] = map[status] || ['未知', 'warn'];
    return pill(text, kind);
  }

  // 作业按钮 + 日志窗:同名作业刚跑完时,重取数会重建这一行,日志尾要能贴回去
  const _logCache = {};

  function jobBox(name, opts) {
    opts = opts || {};
    const bar = el('span', { class: 'job' });
    const btn = el('button', { class: 'btn', text: opts.label || '重跑' });
    const state = el('span', { class: 'muted jobstate', text: '' });
    const log = el('pre', { class: 'log', text: _logCache[name] || '' });
    log.style.display = (_logCache[name] ? 'block' : 'none');

    function render(st) {
      if (st.log_tail) {
        _logCache[name] = st.log_tail;
        log.textContent = st.log_tail;
        log.style.display = 'block';
      }
      if (st.running) { state.textContent = '运行中…'; btn.disabled = true; }
      else if (st.exit_code === null || st.exit_code === undefined) {
        state.textContent = ''; btn.disabled = false;
      } else {
        state.textContent = st.exit_code === 0 ? '上次成功' : '上次失败(exit ' + st.exit_code + ')';
        btn.disabled = false;
      }
    }

    async function poll() {
      let st;
      try { st = await getJSON('/api/jobs/' + encodeURIComponent(name)); }
      catch (e) { finish(e.message); return; }
      render(st);
      if (st.running) setTimeout(poll, 1200);
      else finish(null);
    }

    function finish(err) {
      if (err) state.textContent = '状态读取失败:' + err;
      if (opts.onDone) opts.onDone();
    }

    btn.addEventListener('click', async () => {
      if (opts.heavy && !confirm(opts.heavyText || '这是分钟级作业,确定现在跑?')) return;
      try {
        await getJSON('/api/jobs/' + encodeURIComponent(name), { method: 'POST' });
      } catch (e) {
        state.textContent = e.message;
        return;
      }
      btn.disabled = true;
      state.textContent = '已发起…';
      setTimeout(poll, 600);
    });

    // 初始状态
    getJSON('/api/jobs/' + encodeURIComponent(name)).then(render).catch(() => {});
    bar.appendChild(btn);
    bar.appendChild(state);
    bar.appendChild(log);
    return bar;
  }

  // 验收四页的子导航:挂到 <main> 顶部(#subnav 存在就替换它)
  function subnav(active) {
    const items = [
      ['acceptance', '总览', '/acceptance'],
      ['eval', '评测详情', '/acceptance/eval'],
      ['data', '数据产物', '/acceptance/data'],
      ['errors', '错例复核', '/acceptance/errors'],
    ];
    const box = el('div', { class: 'tabs' });
    items.forEach(([k, text, href]) => {
      const btn = el('button', { class: k === active ? 'on' : '', text });
      btn.addEventListener('click', () => { location.href = href; });
      box.appendChild(btn);
    });
    const host = document.getElementById('subnav');
    if (host) host.replaceWith(box);
    else {
      const main = document.querySelector('main');
      if (main) main.insertBefore(box, main.firstChild.nextSibling);
    }
    return box;
  }

  return { el, getJSON, pill, fmt, fmtBytes, statusPill, jobBox, subnav };
})();
