// ウラガワ共通：各機能のページに「トップへ戻る」ボタンを出す
(() => {
  const base = new URL('../', document.currentScript.src).href; // uragawa/
  const add = () => {
    if (document.getElementById('u-home')) return;
    const st = document.createElement('style');
    st.textContent = `#u-home{position:fixed;z-index:2147483000;top:calc(env(safe-area-inset-top) + 8px);left:8px;
      display:flex;align-items:center;gap:4px;height:30px;padding:0 11px 0 9px;border-radius:15px;
      background:rgba(5,8,11,.72);border:1px solid rgba(95,245,208,.45);color:#5ff5d0;
      font:500 12px/1 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;letter-spacing:.08em;text-decoration:none;
      -webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);-webkit-tap-highlight-color:transparent}
      #u-home:active{opacity:.6}`;
    document.head.appendChild(st);
    const a = document.createElement('a');
    a.id = 'u-home';
    a.href = base;
    a.textContent = '‹ トップ';
    a.setAttribute('aria-label', 'ウラガワのトップに戻る');
    document.body.appendChild(a);
    if (!document.querySelector('link[rel="manifest"]')) {
      const m = document.createElement('link');
      m.rel = 'manifest'; m.href = base + 'manifest.webmanifest';
      document.head.appendChild(m);
    }
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
