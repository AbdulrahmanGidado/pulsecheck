// widget.js — PulseCheck Embed Widget
// Businesses paste one <script> tag and this file does the rest.
// It injects a floating feedback button into their page,
// collects text + star rating, and sends it to the PulseCheck API.

(function () {

  // ── CONFIG ──
  // Read the workspace ID from the script tag's data attribute.
  // e.g. <script src="widget.js" data-workspace="acme-corp"></script>
  const script    = document.currentScript;
  const workspace = script ? script.getAttribute("data-workspace") || "default" : "default";
  const API_URL   = "https://web-production-14e27a.up.railway.app/analyze";

  // ── STYLES ──
  // Inject CSS directly into the page so the widget works on any site.
  const styles = `
    #pc-widget-btn {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 99999;
      background: linear-gradient(135deg, #3b82f6, #a855f7);
      color: #fff;
      border: none;
      border-radius: 50px;
      padding: 12px 20px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
      display: flex;
      align-items: center;
      gap: 8px;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    #pc-widget-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(108, 99, 255, 0.5);
    }
    #pc-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      z-index: 99998;
      display: none;
      align-items: flex-end;
      justify-content: flex-end;
      padding: 24px;
    }
    #pc-overlay.open {
      display: flex;
    }
    #pc-panel {
      background: #0d1018;
      border: 1px solid #1a1e2e;
      border-radius: 20px;
      padding: 24px;
      width: 100%;
      max-width: 360px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      box-shadow: 0 20px 60px rgba(0,0,0,0.5);
      animation: pc-slide-up 0.3s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    @keyframes pc-slide-up {
      from { opacity: 0; transform: translateY(20px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    #pc-panel h3 {
      font-size: 16px;
      font-weight: 700;
      color: #f0f2f5;
      margin: 0 0 4px;
    }
    #pc-panel p {
      font-size: 13px;
      color: #5a6280;
      margin: 0 0 20px;
    }
    .pc-label {
      font-size: 11px;
      font-weight: 600;
      color: #3a4060;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }
    .pc-stars {
      display: flex;
      gap: 6px;
      margin-bottom: 16px;
    }
    .pc-star {
      font-size: 24px;
      cursor: pointer;
      color: #1a1e2e;
      transition: color 0.15s, transform 0.1s;
      line-height: 1;
    }
    .pc-star.active {
      color: #eab308;
    }
    .pc-star:hover {
      transform: scale(1.15);
    }
    #pc-text {
      width: 100%;
      min-height: 90px;
      background: #12151e;
      border: 1px solid #1a1e2e;
      border-radius: 10px;
      padding: 12px 14px;
      color: #e2e6f0;
      font-family: inherit;
      font-size: 13px;
      line-height: 1.6;
      resize: none;
      outline: none;
      box-sizing: border-box;
      margin-bottom: 16px;
      transition: border-color 0.2s;
    }
    #pc-text:focus {
      border-color: #3b82f6;
    }
    #pc-text::placeholder {
      color: #2e3450;
    }
    #pc-submit {
      width: 100%;
      padding: 12px;
      background: linear-gradient(135deg, #3b82f6, #a855f7);
      color: #fff;
      border: none;
      border-radius: 10px;
      font-family: inherit;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
    }
    #pc-submit:hover   { opacity: 0.88; }
    #pc-submit:disabled { opacity: 0.4; cursor: not-allowed; }
    #pc-close {
      position: absolute;
      top: 16px;
      right: 16px;
      background: none;
      border: none;
      color: #3a4060;
      font-size: 18px;
      cursor: pointer;
      line-height: 1;
      padding: 4px;
    }
    #pc-close:hover { color: #e2e6f0; }
    #pc-success {
      text-align: center;
      padding: 20px 0;
      display: none;
    }
    #pc-success .pc-tick {
      font-size: 40px;
      margin-bottom: 12px;
    }
    #pc-success h4 {
      font-size: 16px;
      font-weight: 700;
      color: #f0f2f5;
      margin: 0 0 6px;
    }
    #pc-success p {
      font-size: 13px;
      color: #5a6280;
      margin: 0;
    }
  `;

  // ── INJECT STYLES ──
  const styleEl = document.createElement("style");
  styleEl.textContent = styles;
  document.head.appendChild(styleEl);

  // ── BUILD HTML ──
  // Create the floating button
  const btn = document.createElement("button");
  btn.id = "pc-widget-btn";
  btn.innerHTML = `💬 Share Feedback`;
  document.body.appendChild(btn);

  // Create the overlay + panel
  const overlay = document.createElement("div");
  overlay.id = "pc-overlay";
  overlay.innerHTML = `
    <div id="pc-panel" style="position:relative">
      <button id="pc-close">✕</button>
      <h3>Share your feedback</h3>
      <p>Your thoughts help us improve.</p>

      <div id="pc-form">
        <div class="pc-label">Your rating</div>
        <div class="pc-stars">
          <span class="pc-star" data-val="1">★</span>
          <span class="pc-star" data-val="2">★</span>
          <span class="pc-star" data-val="3">★</span>
          <span class="pc-star" data-val="4">★</span>
          <span class="pc-star" data-val="5">★</span>
        </div>

        <div class="pc-label">Your feedback</div>
        <textarea id="pc-text" placeholder="Tell us what you think…"></textarea>

        <button id="pc-submit">Submit Feedback</button>
      </div>

      <div id="pc-success">
        <div class="pc-tick">✅</div>
        <h4>Thank you!</h4>
        <p>Your feedback has been received.</p>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  // ── STAR RATING LOGIC ──
  let selectedRating = 0;
  const stars = overlay.querySelectorAll(".pc-star");

  stars.forEach(star => {
    // Hover: highlight up to hovered star
    star.addEventListener("mouseenter", () => {
      const val = parseInt(star.getAttribute("data-val"));
      stars.forEach(s => {
        s.classList.toggle("active", parseInt(s.getAttribute("data-val")) <= val);
      });
    });

    // Mouse leave: revert to selected rating
    star.addEventListener("mouseleave", () => {
      stars.forEach(s => {
        s.classList.toggle("active", parseInt(s.getAttribute("data-val")) <= selectedRating);
      });
    });

    // Click: lock in the rating
    star.addEventListener("click", () => {
      selectedRating = parseInt(star.getAttribute("data-val"));
      stars.forEach(s => {
        s.classList.toggle("active", parseInt(s.getAttribute("data-val")) <= selectedRating);
      });
    });
  });

  // ── OPEN / CLOSE ──
  btn.addEventListener("click", () => {
    overlay.classList.add("open");
  });

  overlay.querySelector("#pc-close").addEventListener("click", close);
  overlay.addEventListener("click", e => {
    if (e.target === overlay) close();
  });

  function close() {
    overlay.classList.remove("open");
    // Reset form after closing
    setTimeout(reset, 300);
  }

  function reset() {
    overlay.querySelector("#pc-text").value = "";
    selectedRating = 0;
    stars.forEach(s => s.classList.remove("active"));
    overlay.querySelector("#pc-form").style.display = "block";
    overlay.querySelector("#pc-success").style.display = "none";
    overlay.querySelector("#pc-submit").disabled = false;
    overlay.querySelector("#pc-submit").textContent = "Submit Feedback";
  }

  // ── SUBMIT ──
  overlay.querySelector("#pc-submit").addEventListener("click", async () => {
    const text   = overlay.querySelector("#pc-text").value.trim();
    const submit = overlay.querySelector("#pc-submit");

    if (!text) {
      overlay.querySelector("#pc-text").style.borderColor = "#ef4444";
      return;
    }

    submit.disabled = true;
    submit.textContent = "Analyzing…";

    try {
      await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text:   text,
          mode:   "smart",
          source: `widget-${workspace}`,
        })
      });

      // Show success state
      overlay.querySelector("#pc-form").style.display = "none";
      overlay.querySelector("#pc-success").style.display = "block";

      // Auto close after 2.5 seconds
      setTimeout(close, 2500);

    } catch (e) {
      submit.disabled = false;
      submit.textContent = "Try again";
      console.error("PulseCheck widget error:", e);
    }
  });

})();