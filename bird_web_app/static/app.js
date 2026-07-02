const form = document.querySelector("#uploadForm");
const input = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const button = document.querySelector("#predictButton");
const statusText = document.querySelector("#statusText");
const previewImage = document.querySelector("#previewImage");
const imageFrame = document.querySelector(".image-frame");
const speciesName = document.querySelector("#speciesName");
const description = document.querySelector("#description");
const sourceLine = document.querySelector("#sourceLine");
const confidence = document.querySelector("#confidence");
const modelName = document.querySelector("#modelName");
const accuracy = document.querySelector("#accuracy");
const topList = document.querySelector("#topList");

let objectUrl = null;

function setStatus(message) {
  statusText.textContent = message;
}

function showPreview(file) {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }
  objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  imageFrame.classList.add("has-image");
}

function renderTopList(items) {
  topList.innerHTML = "";
  for (const item of items) {
    const row = document.createElement("li");

    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = item.rank;

    const name = document.createElement("span");
    name.className = "top-name";
    name.textContent = item.display_name || item.name;

    const score = document.createElement("span");
    score.className = "top-confidence";
    score.textContent = `${item.confidence_percent}%`;

    row.append(rank, name, score);
    topList.appendChild(row);
  }
}

function renderSourceLine(data) {
  if (data.description_url) {
    sourceLine.innerHTML = "";
    const prefix = document.createTextNode("说明来源：");
    const link = document.createElement("a");
    link.href = data.description_url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = data.description_source;
    sourceLine.append(prefix, link);
    return;
  }
  sourceLine.textContent = `说明来源：${data.description_source}`;
}

function renderResult(data) {
  speciesName.textContent = data.prediction.display_name || data.prediction.name;
  description.textContent = data.description;
  renderSourceLine(data);
  confidence.textContent = `${data.prediction.confidence_percent}%`;
  modelName.textContent = data.model;
  accuracy.textContent = `${data.test_top1_percent}%`;
  renderTopList(data.top5);
}

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  showPreview(file);
  setStatus(`已选择：${file.name}`);
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) {
    return;
  }
  input.files = event.dataTransfer.files;
  showPreview(file);
  setStatus(`已选择：${file.name}`);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files?.[0];
  if (!file) {
    setStatus("请先选择一张图片。");
    return;
  }

  button.disabled = true;
  setStatus("正在识别，首次加载模型可能需要一些时间。");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "识别失败");
    }
    renderResult(data);
    setStatus(`完成：${data.prediction.display_name || data.prediction.name}`);
  } catch (error) {
    setStatus(error.message || "识别失败，请换一张图片再试。");
  } finally {
    button.disabled = false;
  }
});
