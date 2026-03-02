<script>
/* ── Parallax isolines ── */
(function () {
  const svgs = document.querySelectorAll('.landing .isoline-svg');
  if (!svgs.length) return;

  // Pandoc strips preserveAspectRatio — restore it so isolines fill their containers
  svgs.forEach(svg => svg.setAttribute('preserveAspectRatio', 'xMidYMid slice'));

  // Detect current color mode
  const isDark = document.body.classList.contains('quarto-dark');

  // Remove opposite-contrast shadow layers:
  // Dark mode: remove rgba(0,0,0,...) shadows (black invisible on dark bg)
  // Light mode: remove rgba(255,255,255,...) lines (white invisible on light bg)
  const removePrefix = isDark ? 'rgba(0,0,0' : 'rgba(255,255,255';

  svgs.forEach(svg => {
    svg.querySelectorAll('.iso-layer').forEach(layer => {
      const stroke = layer.getAttribute('stroke') || '';
      if (stroke.startsWith(removePrefix)) {
        layer.remove();
        return;
      }
      // Halve stroke width for crisper lines at scale
      const sw = parseFloat(layer.getAttribute('stroke-width')) || 0.5;
      layer.setAttribute('stroke-width', (sw * 0.6).toFixed(2));
    });
  });

  // Watch for theme toggle — removed SVG layers can't be restored, so reload
  const observer = new MutationObserver(function (mutations) {
    for (const m of mutations) {
      if (m.type === 'attributes' && m.attributeName === 'class') {
        const wasDark = isDark;
        const nowDark = document.body.classList.contains('quarto-dark');
        if (wasDark !== nowDark) {
          observer.disconnect();
          location.reload();
          return;
        }
      }
    }
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });

  const entries = [];
  svgs.forEach(svg => {
    const container = svg.parentElement;
    if (!container) return;
    svg.querySelectorAll('.iso-layer').forEach(layer => {
      const speed = parseFloat(layer.dataset.speed) || 0;
      const existingTransform = layer.getAttribute('transform') || '';
      const m = existingTransform.match(/translate\(([\d.-]+),([\d.-]+)\)/);
      const shadowX = m ? parseFloat(m[1]) : 0;
      const shadowY = m ? parseFloat(m[2]) : 0;
      entries.push({ layer, container, speed, shadowX, shadowY });
    });
  });

  if (!entries.length) return;
  let ticking = false;

  function update() {
    const vh = window.innerHeight;
    entries.forEach(({ layer, container, speed, shadowX, shadowY }) => {
      const rect = container.getBoundingClientRect();
      const centerOffset = (rect.top + rect.height / 2 - vh / 2) / vh;
      const svg = layer.closest('svg');
      const vbH = svg ? parseFloat((svg.getAttribute('viewBox') || '').split(' ')[3]) || 800 : 800;
      const shift = centerOffset * speed * vbH * 4;
      layer.setAttribute('transform',
        'translate(' + shadowX + ',' + (shadowY + shift).toFixed(1) + ')');
    });
    ticking = false;
  }

  window.addEventListener('scroll', function () {
    if (!ticking) { requestAnimationFrame(update); ticking = true; }
  }, { passive: true });
  update();
})();
</script>
