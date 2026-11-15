/* 后台各页共用导航:自带样式注入,挂在样式内联的旧页上也不打架。 */
(function () {
  const CSS = `
  .admin-nav {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 10px 16px; background: #1c1c1c;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  .admin-nav a {
    color: #f6e8cc; text-decoration: none; font-size: 13px;
    padding: 5px 12px; border: 1px solid #6b5335;
  }
  .admin-nav a:hover { background: #2c2c2c; }
  .admin-nav a.active { background: #f7941d; color: #1c1c1c; border-color: #f7941d; font-weight: 600; }
  .admin-nav .spacer { flex: 1; }
  .admin-nav .home { border-color: #f7941d; color: #f7941d; }
  `;

  function mountAdminNav(active) {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    const links = [
      ['admin', '后台首页', '/admin'],
      ['kb', '知识库', '/kb'],
      ['rag-eval', 'RAG 评估', '/rag-eval'],
      ['review', '飞轮待审', '/review'],
      ['observability', '观测与成本', '/observability'],
      ['topics', '主题分布', '/topics'],
      ['acceptance', '分类器验收', '/acceptance'],
      ['chat', '聊天页', '/'],
    ];
    const nav = document.createElement('nav');
    nav.className = 'admin-nav';
    links.forEach(function (item) {
      const a = document.createElement('a');
      a.href = item[2];
      a.textContent = item[1];
      if (item[0] === active) a.className = 'active';
      nav.appendChild(a);
    });

    const host = document.getElementById('admin-nav');
    if (host) {
      host.replaceWith(nav);
    } else {
      document.body.prepend(nav);
    }
    return nav;
  }

  window.mountAdminNav = mountAdminNav;
})();
