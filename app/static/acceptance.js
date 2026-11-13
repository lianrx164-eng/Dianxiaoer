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
    (children || []).forEach((c) => node.appendChild(c));
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

  return { el, getJSON, pill, fmt };
})();
