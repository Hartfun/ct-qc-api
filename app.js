const API = "https://ct-qc-ml.onrender.com";

// Wake API on page load
fetch(API + "/health")
  .then(r => r.json())
  .then(() => {
    document.getElementById("dot").classList.add("online");
    document.getElementById("api-status").textContent = "API Online — ct-qc-ml.onrender.com";
  })
  .catch(() => {
    document.getElementById("api-status").textContent = "API waking up... (~30s)";
  });

function g(id) {
  return document.getElementById(id).value;
}

async function predict() {
  const btn = document.getElementById("btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analysing...';

  const body = {
    "serial No": g("serial_No"),
    "Date": g("Date"),
    "Slice thickness 1.5": parseFloat(g("st15")),
    "Slice thickness 5": parseFloat(g("st5")),
    "Slice thickness 10": parseFloat(g("st10")),
    "KV accuracy 80": parseFloat(g("kv80")),
    "KV accuracy 110": parseFloat(g("kv110")),
    "KV accuracy 130": parseFloat(g("kv130")),
    "Accuracy Timer 0.8": parseFloat(g("t08")),
    "Accuracy Timer 1": parseFloat(g("t1")),
    "Accuracy Timer 1.5": parseFloat(g("t15")),
    "Radiation Dose Test (Head) 21.50": parseFloat(g("dhead")),
    "Radiation Dose Test (Body) 10.60": parseFloat(g("dbody")),
    "Radiation Leakage Levels (Front)": parseFloat(g("lf")),
    "Radiation Leakage Levels (Back)": parseFloat(g("lb")),
    "Radiation Leakage Levels (Left)": parseFloat(g("ll")),
    "Radiation Leakage Levels (Right)": parseFloat(g("lr"))
  };

  try {
    const res = await fetch(API + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const d = await res.json();
    showResult(d);
  } catch(e) {
    const rb = document.getElementById("result-box");
    rb.style.display = "block";
    rb.innerHTML = `<div class="result fail"><h2>❌ Connection Error</h2><p>${e.message}</p></div>`;
  }

  btn.disabled = false;
  btn.innerHTML = "🔍 Run QC Analysis";
}

function showResult(d) {
  const rb = document.getElementById("result-box");
  rb.style.display = "block";

  const cls   = d.overall_acceptance ? (d.anomaly_detected ? "anomaly" : "pass") : "fail";
  const icon  = d.overall_acceptance ? (d.anomaly_detected ? "⚠️" : "✅") : "❌";
  const title = d.overall_acceptance
    ? (d.anomaly_detected ? "ACCEPTED — ML Anomaly Flagged" : "ACCEPTED — All Clear")
    : "REJECTED — Parameter Failure";

  rb.innerHTML = `
    <div class="result ${cls}">
      <h2>${icon} ${title}</h2>
      <div class="scores">
        <div class="score-box"><div class="val">${d.iso_score.toFixed(4)}</div><div class="lbl">ISO Score</div></div>
        <div class="score-box"><div class="val">${d.lof_score.toFixed(4)}</div><div class="lbl">LOF Score</div></div>
        <div class="score-box"><div class="val">${d.ensemble_score.toFixed(4)}</div><div class="lbl">Ensemble Score</div></div>
        <div class="score-box"><div class="val">${d.threshold.toFixed(5)}</div><div class="lbl">Threshold</div></div>
        <div class="score-box"><div class="val">${d.leakage_max} mR/hr</div><div class="lbl">Leakage Max ${d.leakage_pass ? "✅" : "❌"}</div></div>
      </div>
    </div>`;

  const tb = document.getElementById("table-body");
  document.getElementById("table-box").style.display = "block";
  tb.innerHTML = Object.entries(d.parameter_breakdown).map(([k, v]) => `
    <tr>
      <td>${k}</td>
      <td><b>${v.value}</b></td>
      <td>${v.spec}</td>
      <td>±${v.tolerance}</td>
      <td>${v.pct_deviation.toFixed(3)}%</td>
      <td><span class="badge ${v.pass ? 'pass' : 'fail'}">${v.pass ? "✅ PASS" : "❌ FAIL"}</span></td>
    </tr>`).join("");
}
