/* ═══════════════════════════════════════════
   STYLE AI — Frontend JavaScript (SPA)
═══════════════════════════════════════════ */

// ─── State ───
let selectedGender = 'male';
let selectedFile = null;
let currentMode = 'upload';
let cameraStream = null;
let liveInterval = null;
let lastResults = null;

// Avatar State
let avatarSelectedTone = null;
let avatarSelectedToneHex = null;
let avatarSelectedToneName = null;

// ─── Proxy Image URL helper ───
function pImg(q, w = 400, h = 300) {
  return `/proxy-image?q=${encodeURIComponent(q)}&w=${w}&h=${h}&cb=${Date.now()}`;
}

// ─── Category image queries ───
const CQ = {
  shirt: { m: 'mens shirt product photography isolated white background', f: 'womens top blouse product photography isolated white background' },
  pants: { m: 'mens chinos trousers product photography isolated white background', f: 'womens trousers pants product photography isolated white background' },
  shoes: { m: 'mens footwear shoes product photography isolated white background', f: 'womens footwear shoes product photography isolated white background' },
  acc: { m: 'mens watch accessories product photography isolated white background', f: 'womens jewelry accessories product photography isolated white background' },
  full: { m: 'mens fashion clothing outfit set product photography isolated', f: 'womens fashion clothing outfit set product photography isolated' },
};

// ─── Page Navigation (SPA) ───
function navigateTo(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const page = document.getElementById('page-' + pageId);
  if (page) { page.classList.add('active'); window.scrollTo(0, 0); }
  document.querySelectorAll('.nav-links a[data-page]').forEach(a =>
    a.classList.toggle('nav-active', a.dataset.page === pageId)
  );
  // Trigger GSAP stagger animation for newly visible page
  if (typeof window._gsapPageAnim === 'function') window._gsapPageAnim(pageId);
}


// ─── Tab Switching ───
function switchTab(tab) {
  currentMode = tab;
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab)
  );
  document.getElementById('uploadPanel').style.display = tab === 'upload' ? '' : 'none';
  document.getElementById('cameraPanel').style.display = tab === 'camera' ? '' : 'none';
  document.getElementById('avatarPanel').style.display = tab === 'avatar' ? '' : 'none';

  document.getElementById('analyzeBtn').style.display = tab === 'upload' ? '' : 'none';
  document.getElementById('cameraBtnCapture').style.display = tab === 'camera' ? '' : 'none';
  document.getElementById('avatarBtnGenerate').style.display = tab === 'avatar' ? '' : 'none';

  if (tab === 'camera') startCamera(); else stopCamera();
}

// ─── Gender ───
function setGender(gender, el) {
  selectedGender = gender;
  document.querySelectorAll('.gender-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
}

// ─── Preferences Toggler ───
function togglePreferences() {
  const btn = document.getElementById('prefToggleBtn');
  const content = document.getElementById('preferencesContent');
  btn.classList.toggle('open');
  if (content.style.display === 'none') {
    content.style.display = 'block';
  } else {
    content.style.display = 'none';
  }
}

// ─── Avatar Builder ───
function selectAvatarTone(toneIdx, hexColor, toneName) {
  avatarSelectedTone = toneIdx;
  avatarSelectedToneHex = hexColor;
  avatarSelectedToneName = toneName;

  document.querySelectorAll('.avatar-tone-option').forEach(el =>
    el.classList.toggle('active', parseInt(el.dataset.tone) === toneIdx)
  );

  const label = document.getElementById('avatarToneLabel');
  label.innerHTML = `Selected Skin Tone: <span>${toneName}</span> (MST-${toneIdx})`;

  document.getElementById('avatarBtnGenerate').disabled = false;
}

async function generateAvatarStyle() {
  if (!avatarSelectedTone) return;

  const btn = document.getElementById('avatarBtnGenerate');
  const orgHtml = btn.innerHTML;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating styling...';
  btn.disabled = true;

  try {
    showToast('Consulting HuggingFace LLM with your avatar traits...', 'info');

    // We send a mock image request to the server, but pass a flag indicating it's an avatar run
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        is_avatar: true,
        gender: selectedGender,
        avatar_tone_index: avatarSelectedTone, // 1 through 9
        pref_height: document.getElementById('prefHeight').value.trim(),
        pref_budget: document.getElementById('prefBudget').value,
        pref_brands: document.getElementById('prefBrands').value.trim()
      })
    });

    if (!res.ok) throw new Error((await res.json()).error || 'Analysis failed');
    const data = await res.json();
    lastResults = data;
    renderResults(data);
  } catch (err) {
    showToast('Error: ' + err.message, 'error');
  } finally {
    btn.innerHTML = orgHtml;
    btn.disabled = false;
  }
}

// ─── Camera ───
async function startCamera() {
  const st = document.getElementById('cameraStatus');
  st.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Starting camera…';
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    const video = document.getElementById('cameraFeed');
    video.srcObject = cameraStream;
    await video.play();
    document.getElementById('cameraBtnCapture').disabled = false;
    st.innerHTML = '<i class="fas fa-circle live-indicator"></i> Detecting skin tone live…';
    liveInterval = setInterval(runLiveDetect, 2500);
  } catch (err) {
    st.textContent = '❌ Camera access denied. Check browser permissions.';
    showToast('❌ Camera access denied. Allow permission in your browser.', 'error');
  }
}

function stopCamera() {
  if (liveInterval) { clearInterval(liveInterval); liveInterval = null; }
  if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
  const badge = document.getElementById('liveToneBadge');
  if (badge) badge.style.display = 'none';
}

function grabFrame() {
  const video = document.getElementById('cameraFeed');
  const canvas = document.getElementById('cameraCanvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.8));
}

function blobToBase64(blob) {
  return new Promise(res => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(',')[1]);
    r.readAsDataURL(blob);
  });
}

async function runLiveDetect() {
  if (!cameraStream) return;
  try {
    const blob = await grabFrame();
    const b64 = await blobToBase64(blob);
    const resp = await fetch('/detect-live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 })
    });
    const data = await resp.json();
    if (data.success) {
      const st = data.skin_tone;
      const badge = document.getElementById('liveToneBadge');
      document.getElementById('liveToneText').textContent = `${st.name} Skin Tone`;
      badge.style.background = st.hex;
      badge.style.color = (st.r * 0.299 + st.g * 0.587 + st.b * 0.114) < 128 ? '#fff' : '#111';
      badge.style.display = 'flex';
      document.getElementById('cameraStatus').innerHTML = '<i class="fas fa-check-circle" style="color:#4ade80"></i> Face detected! Ready to capture.';
    } else if (data.error && data.error.includes('No face')) {
      document.getElementById('liveToneBadge').style.display = 'none';
      document.getElementById('cameraStatus').innerHTML = '<i class="fas fa-user" style="color:var(--text-muted)"></i> Waiting for a clear face...';
    }
  } catch (e) { /* silent */ }
}

async function captureAndAnalyze() {
  if (!cameraStream) return;
  const blob = await grabFrame();
  doAnalyze(blob, 'capture.jpg');
}

// ─── Drag & Drop ───
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
dropZone.addEventListener('click', e => {
  if (!e.target.closest('.dz-preview') && !e.target.closest('.btn-browse')) fileInput.click();
});

function handleFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
  if (!allowed.includes(file.type)) { showToast('❌ Invalid file type. Use PNG, JPG, JPEG, GIF or WEBP.', 'error'); return; }
  if (file.size > 10 * 1024 * 1024) { showToast('❌ File too large. Max 10MB.', 'error'); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('dzInner').style.display = 'none';
    document.getElementById('dzPreview').style.display = 'block';
    document.getElementById('analyzeStatus').textContent = 'Ready to analyze!';
    document.getElementById('analyzeBtn').disabled = false;
  };
  reader.readAsDataURL(file);
}

function analyzePhoto() { if (selectedFile) doAnalyze(selectedFile, selectedFile.name); }

function resetAndAnalyze() {
  selectedFile = null;
  if (fileInput) fileInput.value = '';
  document.getElementById('dzInner').style.display = '';
  document.getElementById('dzPreview').style.display = 'none';
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('progressWrap').style.display = 'none';
  stopCamera();
  switchTab('upload');
  navigateTo('analyzer');
}

// ─── Core Analyze ───
async function doAnalyze(file, filename) {
  const isCamera = currentMode === 'camera';
  const btnId = isCamera ? 'cameraBtnCapture' : 'analyzeBtn';
  const btn = document.getElementById(btnId);
  const origLabel = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing…';
  if (!isCamera) {
    document.getElementById('analyzeStatus').textContent = 'Analyzing your style…';
    document.querySelector('.analyzing-ring').classList.add('analyzing');
  }
  const pw = document.getElementById('progressWrap');
  const pb = document.getElementById('progressBar');
  const pl = document.getElementById('progressLabel');
  pw.style.display = 'block';
  animateProgress(pb, pl);

  try {
    const formData = new FormData();
    formData.append('file', file, filename || 'photo.jpg');
    formData.append('gender', selectedGender);
    formData.append('pref_height', (document.getElementById('prefHeight')?.value || '').trim());
    formData.append('pref_budget', document.getElementById('prefBudget')?.value || '');
    formData.append('pref_brands', (document.getElementById('prefBrands')?.value || '').trim());
    formData.append('pref_occasion', document.getElementById('prefOccasion')?.value || 'casual');
    formData.append('pref_weather', document.getElementById('prefWeather')?.value || '');
    const response = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await response.json();

    if (!data.success) {
      btn.disabled = false; btn.innerHTML = origLabel; pw.style.display = 'none';
      if (!isCamera) document.querySelector('.analyzing-ring').classList.remove('analyzing');

      // ─── Special gender mismatch error ───
      if (data.gender_mismatch) {
        showGenderMismatchError(data.detected_gender, data.selected_gender);
      } else {
        showToast('❌ ' + (data.error || 'Analysis failed'), 'error');
      }
      return;
    }

    pb.style.width = '100%'; pl.textContent = 'Analysis complete! ✓';
    btn.disabled = false; btn.innerHTML = origLabel;
    if (!isCamera) document.querySelector('.analyzing-ring').classList.remove('analyzing');
    setTimeout(() => { pw.style.display = 'none'; renderResults(data); }, 600);
  } catch (err) {
    showToast('❌ Network error. Is the Flask server running?', 'error');
    btn.disabled = false; btn.innerHTML = origLabel; pw.style.display = 'none';
  }
}

function animateProgress(bar, label) {
  const steps = [
    { pct: 15, text: 'Uploading image…', delay: 100 },
    { pct: 35, text: 'Detecting skin tone…', delay: 800 },
    { pct: 60, text: 'Connecting to HuggingFace AI…', delay: 1800 },
    { pct: 85, text: 'Generating recommendations…', delay: 3500 },
    { pct: 95, text: 'Almost done…', delay: 5500 },
  ];
  bar.style.width = '0%';
  steps.forEach(s => setTimeout(() => { bar.style.width = s.pct + '%'; label.textContent = s.text; }, s.delay));
}

// ─── Render Results ───
function renderResults(data) {
  lastResults = data;
  const { skin_tone, recommendations, products } = data;

  // Skin tone card
  document.getElementById('toneSwatch').style.background = skin_tone.hex;
  document.getElementById('toneName').textContent = skin_tone.name + ' Skin Tone';
  document.getElementById('toneRgb').textContent = `RGB: ${skin_tone.r}, ${skin_tone.g}, ${skin_tone.b}`;
  document.getElementById('toneHex').textContent = `HEX: ${skin_tone.hex.toUpperCase()}`;

  // Gender badge in results
  const genderBadge = document.getElementById('genderUsedBadge');
  const gIcon = selectedGender === 'male' ? 'fa-mars' : 'fa-venus';
  const gLabel = selectedGender === 'male' ? 'Male' : 'Female';
  genderBadge.innerHTML = `<i class="fas ${gIcon}"></i> ${gLabel} Profile`;
  genderBadge.className = `gender-used-badge gender-${selectedGender}`;

  // Define gender key early so both Avatar & Photo logic can use it
  const gKey = selectedGender === 'male' ? 'm' : 'f';

  // Avatar vs Photo Preview logic
  if (data.is_avatar) {
    document.getElementById('outfitPreviewSection').style.display = 'none';
    document.getElementById('avatarTryOnSection').style.display = 'block';

    // Tint the base SVG body to match the selected skin tone
    document.getElementById('avatarBodyPath').setAttribute('fill', skin_tone.hex);

    // Defer the clothing layer rendering so the UI doesn't block while proxying
    setTimeout(() => renderAvatarLayers(products, gKey), 100);
  } else {
    document.getElementById('avatarTryOnSection').style.display = 'none';
    document.getElementById('outfitPreviewSection').style.display = 'block';

    // Outfit preview image (proxied with precise query based on 9-tone scale)
    const outfitQ = `${CQ.full[gKey]} ${skin_tone.name.toLowerCase()} skin tone model`;
    document.getElementById('outfitPreviewImg').src = pImg(outfitQ, 1200, 500);
  }

  // Recommendation cards
  document.getElementById('recsGrid').innerHTML = parseRecommendations(recommendations, skin_tone.hex, gKey);

  // Product cards with preview modal
  renderProducts(products, skin_tone, gKey);

  // Apply 3D Tilt interactive effects
  if (typeof VanillaTilt !== 'undefined') {
    requestAnimationFrame(() => {
      VanillaTilt.init(document.querySelectorAll('.product-card'), {
        max: 8, speed: 400, glare: true, "max-glare": 0.2, scale: 1.02
      });
      VanillaTilt.init(document.querySelectorAll('.rec-card'), {
        max: 5, speed: 400, glare: true, "max-glare": 0.1
      });
    });
  }

  navigateTo('results');
  showToast('✓ Your style profile is ready!', 'success');
}

// ─── Parse Recommendations ───
function parseRecommendations(text, hexColor, gKey) {
  const sections = {
    'DRESS_CODE': { icon: 'fas fa-tags', title: 'Dress Codes' },
    'SUGGESTED_OUTFIT': { icon: 'fas fa-tshirt', title: 'Suggested Outfit' },
    'SHIRT_DETAILS': { icon: 'fas fa-grip-lines', title: 'Shirt / Top Details' },
    'PANT_DETAILS': { icon: 'fas fa-ruler-vertical', title: 'Trousers / Bottom Details' },
    'SHOES_DETAILS': { icon: 'fas fa-shoe-prints', title: 'Footwear Details' },
    'HAIRSTYLE': { icon: 'fas fa-cut', title: 'Hairstyle' },
    'ACCESSORIES': { icon: 'fas fa-gem', title: 'Accessories' },
    'COLOR_PALETTE': { icon: 'fas fa-palette', title: 'Colour Palette' },
    'WHY_IT_WORKS': { icon: 'fas fa-lightbulb', title: 'Why It Works' },
  };
  const keys = Object.keys(sections);
  let cards = '';

  keys.forEach((key, idx) => {
    const regex = new RegExp(key + '[_\\s]*\\n([\\s\\S]*?)(?=' + keys.slice(idx + 1).join('|') + '|$)', 'i');
    const match = text.match(regex);
    if (!match) return;
    const rawContent = match[1].trim();
    const lines = rawContent.split('\n').filter(l => l.trim());
    let bodyHtml = '';

    // No proxy images in rec cards — text-only for clean, relevant display

    if (key === 'COLOR_PALETTE') {
      const colors = [];
      lines.forEach(l => {
        const clean = l.replace(/^[→\-\*•]+\s*/, '').trim();
        if (clean) { const [label, ...rest] = clean.split(':'); colors.push({ label: label.trim(), value: rest.join(':').trim() }); }
      });
      const sw = { Primary: '#2c3e7a', Secondary: '#6b8f71', Accent: '#c8843c' };
      bodyHtml += '<div class="palette-swatches">';
      colors.forEach(c => {
        const bg = hexColor && c.label.toLowerCase() === 'primary' ? hexColor : (sw[c.label] || '#555');
        bodyHtml += `<div class="palette-chip" style="background:${bg}">${c.label}</div>`;
      });
      bodyHtml += '</div>';
      colors.forEach(c => { bodyHtml += `<div class="rec-item" style="margin-top:10px">${c.label}: <strong style="color:var(--cream)">${c.value}</strong></div>`; });
    } else if (key === 'SUGGESTED_OUTFIT') {
      lines.forEach(l => {
        const clean = l.replace(/^[→\-\*•]+\s*/, '').trim();
        if (clean) bodyHtml += `<div class="rec-item">${clean}</div>`;
      });
      bodyHtml += `<button class="love-look-btn" onclick="toggleLoveLook(this)"><i class="far fa-heart"></i> Love This Look</button>`;
    } else {
      lines.forEach(l => {
        const clean = l.replace(/^[→\-\*•]+\s*/, '').trim();
        if (clean) bodyHtml += `<div class="rec-item">${clean}</div>`;
      });
    }

    cards += `
      <div class="rec-card" style="animation:fadeUp .4s ease ${idx * .07}s both">
        <div class="rec-card-header">
          <div class="rec-card-icon"><i class="${sections[key].icon}"></i></div>
          <div class="rec-card-title">${sections[key].title}</div>
        </div>
        <div class="rec-card-body">${bodyHtml || rawContent}</div>
      </div>`;
  });

  return cards || `<div class="rec-card"><div class="rec-card-body">${text}</div></div>`;
}

function toggleLoveLook(btn) {
  const loved = btn.classList.toggle('loved');
  btn.innerHTML = loved ? '<i class="fas fa-heart"></i> Loved! ♥' : '<i class="far fa-heart"></i> Love This Look';
  if (loved) showToast('♥ Added to your loved looks!', 'success');
}

// ─── Render Products ───
const PRODUCT_ICONS = [
  { icon: 'fas fa-tshirt', color: '#c9a96e', label: 'Top / Shirt' },
  { icon: 'fas fa-socks', color: '#6b8f71', label: 'Bottom / Trousers' },
  { icon: 'fas fa-shoe-prints', color: '#7a6abf', label: 'Footwear' },
  { icon: 'fas fa-gem', color: '#bf6a6a', label: 'Accessories' },
  { icon: 'fas fa-vest', color: '#4a9eff', label: 'Full Outfit' },
];

function renderProducts(products, skinTone, gKey) {
  const grid = document.getElementById('productsGrid');
  let html = '';
  products.forEach((p, i) => {
    const meta = PRODUCT_ICONS[i] || PRODUCT_ICONS[0];

    // Extract the primary store domain to force Bing to scrape that exact e-commerce site
    let siteQuery = '';
    if (p.stores[0] && p.stores[0].url) {
      try {
        const domain = new URL(p.stores[0].url).hostname;
        siteQuery = `site:${domain}`;
      } catch (e) { }
    }
    const genderTerm = gKey === 'm' ? 'mens ' : 'womens ';

    // Search for the EXACT product name + SITE DOMAIN + "product isolated"
    const imgQ = `${siteQuery} ${genderTerm} ${p.name} product isolated white background`.trim();
    const imgSrc = pImg(imgQ, 500, 360);

    let storeLinks = '';
    p.stores.forEach(s => {
      storeLinks += `<a href="${s.url}" target="_blank" rel="noopener noreferrer" class="store-link"><span class="store-link-icon">${s.icon}</span><span>${s.name}</span><i class="fas fa-external-link-alt store-link-arrow"></i></a>`;
    });
    const previewData = JSON.stringify({ name: p.name, stores: p.stores }).replace(/'/g, "\\'");

    html += `
      <div class="product-card" style="animation:fadeUp .4s ease ${i * .08}s both">
        <div class="product-img-wrap" style="height: 220px; position: relative; overflow: hidden; background: var(--surface2)">
          <img src="${imgSrc}" alt="${p.name}" loading="lazy" style="width: 100%; height: 100%; object-fit: cover; transition: transform .4s ease;">
          <button class="preview-look-btn" onclick='openPreviewModal(${previewData},"${imgSrc}")' style="position: absolute; bottom: 16px; left: 16px; right: 16px; background: rgba(15,13,11,0.85); backdrop-filter: blur(8px); border: 1px solid var(--gold); color: var(--gold); padding: 10px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: var(--transition);">
            <i class="fas fa-eye"></i> View Look
          </button>
        </div>
        <div class="product-card-body">
          <div class="product-name-mini" style="font-size:0.75rem; color:${meta.color}; margin-bottom:10px; font-weight:700; text-transform:uppercase; display:flex; gap:6px; align-items:center;">
            <i class="${meta.icon}"></i> ${meta.label}
          </div>
          <h4 class="product-name">${p.name}</h4>
          <div class="store-links">${storeLinks}</div>
        </div>
      </div>`;
  });
  grid.innerHTML = html;
}

// ─── Preview Modal ───
function openPreviewModal(product, imgSrc) {
  document.getElementById('modalImg').src = imgSrc;
  document.getElementById('modalTitle').textContent = product.name;
  document.getElementById('modalDesc').textContent = `Browse this style across multiple premium stores below.`;

  let storesHtml = '';
  product.stores.forEach(s => {
    storesHtml += `
      <a href="${s.url}" target="_blank" rel="noopener noreferrer" class="modal-store-btn" style="border-color:${s.color}20;color:${s.color}">
        <span>${s.icon}</span> Shop at ${s.name} <i class="fas fa-external-link-alt"></i>
      </a>`;
  });
  document.getElementById('modalStores').innerHTML = storesHtml;
  document.getElementById('previewModal').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closePreviewModal() {
  document.getElementById('previewModal').classList.remove('open');
  document.body.style.overflow = '';
}

function closeModal(e) {
  if (e.target.id === 'previewModal') closePreviewModal();
}

// ─── Toast ───
function showToast(msg, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  setTimeout(() => toast.classList.remove('show'), 3500);
}

// ─── Gender Mismatch Error Card ───
function showGenderMismatchError(detectedGender, selectedGender) {
  // Remove any existing mismatch banner
  const existing = document.getElementById('genderMismatchBanner');
  if (existing) existing.remove();

  const correctGender = detectedGender; // what AI found
  const wrongGender = selectedGender; // what user picked

  const banner = document.createElement('div');
  banner.id = 'genderMismatchBanner';
  banner.className = 'gender-mismatch-banner';
  banner.innerHTML = `
    <div class="gmb-icon"><i class="fas fa-exclamation-triangle"></i></div>
    <div class="gmb-content">
      <h4>Gender Mismatch Detected</h4>
      <p>
        Our AI detected a <strong>${detectedGender}</strong> face in your photo,
        but you selected <strong>${selectedGender}</strong>.
        Please choose one of the options below to continue.
      </p>
      <div class="gmb-actions">
        <button class="gmb-btn gmb-primary" onclick="resetAndAnalyze()">
          <i class="fas fa-upload"></i> Upload a ${selectedGender} Photo
        </button>
        <button class="gmb-btn gmb-secondary" onclick="fixGenderAndRetry('${correctGender.toLowerCase()}')">
          <i class="fas fa-venus-mars"></i> Switch to ${correctGender} &amp; Retry
        </button>
      </div>
    </div>
    <button class="gmb-close" onclick="this.closest('#genderMismatchBanner').remove()">
      <i class="fas fa-times"></i>
    </button>`;

  // Insert banner at top of the upload form column
  const formCol = document.querySelector('.upload-form-col');
  if (formCol) formCol.insertBefore(banner, formCol.firstChild);
  else document.querySelector('.analyzer-section .container').prepend(banner);

  // Scroll into view
  banner.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function fixGenderAndRetry(gender) {
  // Update gender, remove banner, re-run analysis
  const btn = document.querySelector(`.gender-btn[data-gender="${gender}"]`);
  if (btn) setGender(gender, btn);
  const banner = document.getElementById('genderMismatchBanner');
  if (banner) banner.remove();
  if (selectedFile) doAnalyze(selectedFile, selectedFile.name);
  else showToast('Please upload a photo first', 'error');
}

// ─── Avatar Try-On Helpers ───
function renderAvatarLayers(products, gKey) {
  // Map our 0=Shirt, 1=Pants, 2=Shoes from the backend list
  const layerConfigs = [
    { idx: 0, elId: 'avatarLayerShirt', label: 'Shirt' },
    { idx: 1, elId: 'avatarLayerPants', label: 'Pants' },
    { idx: 2, elId: 'avatarLayerShoes', label: 'Shoes' }
  ];

  const genderTerm = gKey === 'm' ? 'mens ' : 'womens ';

  layerConfigs.forEach(cfg => {
    if (!products[cfg.idx]) return;
    const p = products[cfg.idx];

    let siteQuery = '';
    if (p.stores[0] && p.stores[0].url) {
      try { siteQuery = `site:${new URL(p.stores[0].url).hostname}`; } catch (e) { }
    }

    // We add "transparent PNG" to the query so the scraper tries to find isolated clothing
    const imgQ = `${siteQuery} ${genderTerm} ${p.name} product flat lay transparent PNG`.trim();
    const imgSrc = pImg(imgQ, 400, 400);

    const layer = document.getElementById(cfg.elId);
    if (layer) {
      layer.querySelector('img').src = imgSrc;
      layer.title = `Shop ${p.name}`;
    }
  });
}

function scrollToProduct(productIdx) {
  const card = document.getElementById(`productCard_${productIdx}`);
  if (card) {
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Add a glowing highlight effect
    card.style.transition = 'box-shadow 0.3s ease';
    card.style.boxShadow = '0 0 30px rgba(201, 169, 110, 0.8)';
    setTimeout(() => { card.style.boxShadow = ''; }, 1500);
  }
}


// ─── Animations ───
const _s = document.createElement('style');
_s.textContent = `@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}`;
document.head.appendChild(_s);

// ─── Init ───
window.addEventListener('DOMContentLoaded', async () => {
  navigateTo('home');

  // ── Apply gender coloring to nav ──
  const body = document.body;
  const gender = window.USER_GENDER || 'male';
  body.classList.add(gender === 'female' ? 'gender-female' : 'gender-male');
  selectedGender = gender;
  // Pre-select gender button if present
  document.querySelectorAll('.gender-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.gender === gender);
  });

  // ── Occasion chips wiring ──
  document.querySelectorAll('.occ-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.occ-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const occ = chip.dataset.occ;
      const el = document.getElementById('prefOccasion');
      if (el) el.value = occ;
    });
  });

  // Initialize floating particles (requires tsParticles & preset-stars)
  if (typeof tsParticles !== 'undefined') {
    tsParticles.load("tsparticles", {
      preset: "stars",
      background: { color: "transparent" },
      particles: {
        number: { value: 60 },
        color: { value: ["#c9a96e", "#60a5fa", "#ffffff"] },
        opacity: { value: 0.5, random: true },
        size: { value: 2, random: true },
        move: { enable: true, speed: 0.5, direction: "none" }
      }
    });
  }

  // ── Fetch weather ──
  async function loadWeather() {
    const city = window.USER_CITY;
    if (!city) return;
    try {
      const r = await fetch(`/weather?city=${encodeURIComponent(city)}`);
      const d = await r.json();
      if (!d.success) return;
      // Navbar widget
      const nEmoji = document.getElementById('weatherEmoji');
      const nTemp  = document.getElementById('weatherTemp');
      const nCity  = document.getElementById('weatherCity');
      if (nEmoji) nEmoji.textContent = d.emoji;
      if (nTemp ) nTemp.textContent  = `${d.temperature}°C`;
      if (nCity ) nCity.textContent  = d.city;
      // Analyzer card widget
      const wEmoji = document.getElementById('wcEmoji');
      const wTemp  = document.getElementById('wcTemp');
      const wLoc   = document.getElementById('wcLoc');
      if (wEmoji) wEmoji.textContent = d.emoji;
      if (wTemp ) wTemp.textContent  = `${d.temperature}°C — ${d.description}`;
      if (wLoc  ) wLoc.textContent   = `${d.city}, ${d.country}`;
      // Store for LLM
      const wField = document.getElementById('prefWeather');
      if (wField) wField.value = `${d.temperature}°C, ${d.description}, ${d.condition}, ${d.season} season`;
    } catch(e) { console.log('Weather fetch error:', e); }
  }
  loadWeather();

  // ── AI status check ──
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const el = document.getElementById('navStatus');
    if (data.hf_connected) {
      el.innerHTML = '<span class="status-dot"></span> AI Ready';
      el.style.color = 'var(--success)';
    } else {
      el.innerHTML = '<span class="status-dot" style="background:orange"></span> Fallback Mode';
      el.style.color = 'orange';
    }
  } catch (e) { }
});


// --- GSAP Animations (runs after DOM is ready) ---
(function initGSAP() {
  if (typeof gsap === 'undefined') return;

  gsap.registerPlugin(ScrollTrigger);

  // Helper: animate elements when they scroll into view
  function scrollReveal(selector, vars) {
    gsap.utils.toArray(selector).forEach((el, i) => {
      gsap.from(el, {
        scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none none' },
        ...vars,
        delay: (vars.stagger || 0) * i,
      });
    });
  }

  // -- Home page --
  scrollReveal('.feature-card', { opacity: 0, y: 40, duration: 0.65, stagger: 0.1, ease: 'power3.out' });
  scrollReveal('.stat', { opacity: 0, y: 20, duration: 0.5, stagger: 0.12, ease: 'back.out(1.5)' });

  // -- Hero pipeline steps --
  gsap.from('.hp-step', {
    opacity: 0, y: 20, stagger: 0.15, duration: 0.6, ease: 'power2.out', delay: 0.8
  });
  gsap.from('.hp-arrow', {
    opacity: 0, scaleX: 0, stagger: 0.15, duration: 0.4, ease: 'power2.out', delay: 1
  });

  // -- Hero skin tone cards stagger --
  gsap.from('.hero-card', {
    opacity: 0, x: 30, stagger: 0.12, duration: 0.7, ease: 'power3.out', delay: 0.6
  });

  // -- How It Works: step cards --
  scrollReveal('.step-card', { opacity: 0, y: 35, duration: 0.6, stagger: 0.1, ease: 'power3.out' });
  scrollReveal('.tech-card', { opacity: 0, scale: 0.88, duration: 0.55, stagger: 0.09, ease: 'back.out(1.6)' });

  // -- Analyzer page elements --
  scrollReveal('.gender-selector', { opacity: 0, y: 20, duration: 0.5, ease: 'power2.out' });
  scrollReveal('.mode-tabs', { opacity: 0, y: 15, duration: 0.4, ease: 'power2.out' });
  scrollReveal('.upload-tips-col', { opacity: 0, x: 30, duration: 0.6, ease: 'power3.out' });

  // -- Results page --
  scrollReveal('.tone-result-card', { opacity: 0, y: 24, duration: 0.6, ease: 'power3.out' });
  scrollReveal('.rec-card', { opacity: 0, y: 32, duration: 0.55, stagger: 0.08, ease: 'power3.out' });
  scrollReveal('.product-card', { opacity: 0, y: 28, duration: 0.5, stagger: 0.07, ease: 'power3.out' });

  // -- Re-run animations when navigating between SPA pages --
  window._gsapPageAnim = function (pageId) {
    const sel = {
      home: ['.feature-card', '.stat', '.hero-card', '.hp-step'],
      analyzer: ['.gender-selector', '.mode-tabs', '.upload-tips-col', '.ai-stack-card'],
      results: ['.tone-result-card', '.rec-card', '.product-card'],
      about: ['.step-card', '.tech-card'],
    }[pageId] || [];

    sel.forEach((s, gi) => {
      gsap.utils.toArray(s).forEach((el, i) => {
        gsap.from(el, { opacity: 0, y: 28, duration: 0.5, ease: 'power3.out', delay: gi * 0.05 + i * 0.07 });
      });
    });

    ScrollTrigger.refresh();
  };
})();
