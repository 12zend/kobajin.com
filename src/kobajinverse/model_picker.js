import { DEFAULT_VERSION, MODELS } from "../models.js";
import { $ } from "./dom.js";
import { llm } from "./llm.js";

// Model picker. The mock ignores it; with WebLLM, load only the chosen
// version, when it is first needed (see AGENTS.md).

export function setupModelPicker() {
  const versionSelect = $("model-version");
  const MODEL_KEY = "kobajinverse.model";
  let chosenVersion = DEFAULT_VERSION;
  try { chosenVersion = localStorage.getItem(MODEL_KEY) || DEFAULT_VERSION; } catch { /* default */ }
  versionSelect.innerHTML = MODELS.map((model) => `<option value="${model.version}">${model.version}${model.version === DEFAULT_VERSION ? "（標準）" : ""}</option>`).join("");
  versionSelect.value = MODELS.some((model) => model.version === chosenVersion) ? chosenVersion : DEFAULT_VERSION;
  versionSelect.addEventListener("change", () => {
    try { localStorage.setItem(MODEL_KEY, versionSelect.value); } catch { /* this visit only */ }
  });
  $("model-state").textContent = llm.name === "mock" ? "モック推論" : "ブラウザ内で推論";
  document.querySelectorAll(".mock-only").forEach((el) => { el.hidden = llm.name !== "mock"; });
}
