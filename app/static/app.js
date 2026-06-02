const form = document.getElementById("predict-form");
const input = document.getElementById("image-input");
const previewImg = document.getElementById("preview-img");
const previewEmpty = document.getElementById("preview-empty");
const resultStatus = document.getElementById("result-status");
const resultMeta = document.getElementById("result-meta");
const clearBtn = document.getElementById("clear-btn");
const apiBase = (window.API_BASE_URL || "").replace(/\/$/, "");

function setResult(status, meta) {
  resultStatus.textContent = status;
  resultMeta.textContent = meta || "";
}

function resetUI() {
  input.value = "";
  previewImg.src = "";
  previewImg.style.display = "none";
  previewEmpty.style.display = "grid";
  setResult("Waiting", "");
}

input.addEventListener("change", () => {
  const file = input.files[0];
  if (!file) {
    resetUI();
    return;
  }

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = "block";
  previewEmpty.style.display = "none";
  setResult("Ready", "Click Run Prediction");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) {
    setResult("No file", "Please choose an image first.");
    return;
  }

  setResult("Running...", "Uploading image and waiting for prediction.");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const detail = err.detail || `Request failed (${response.status})`;
      setResult("Error", detail);
      return;
    }

    const data = await response.json();
    const meta = `anomaly_score: ${data.anomaly_score}\nthreshold: ${data.threshold}`;
    setResult(data.prediction, meta);
  } catch (error) {
    setResult("Error", error.message || "Request failed.");
  }
});

clearBtn.addEventListener("click", resetUI);

resetUI();
