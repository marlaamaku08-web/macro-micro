// index.html-ийн AI logic-ийг mock DOM + mock fetch ашиглан шалгана.
// Ажиллуулах: node tests/frontend_regression.js
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const code = scripts[scripts.length - 1][1];

let passed = 0;
let failed = 0;
function check(condition, message) {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.log("   FAIL:", message);
  }
}

async function run(fileName, mode = "ok") {
  const els = {};
  const handlers = {};
  const calls = [];
  const scrolls = [];
  let buttonDuringFetch;

  const el = (id) => els[id] || (els[id] = {
    id,
    style: {},
    innerHTML: id === "ai-predict-button" ? "\n              AI-аар шинжлэх\n            " : "",
    textContent: "",
    value: "x",
    files: [],
    disabled: false,
    addEventListener: (event, fn) => { (handlers[id + event] = handlers[id + event] || []).push(fn); },
    dispatchEvent() {},
    removeAttribute(attr) { this[attr + "Removed"] = true; },
    scrollIntoView(options) { scrolls.push(id + ":" + options.behavior); },
    classList: { add() {}, remove() {}, toggle() {} },
    querySelectorAll: () => [],
    appendChild() {},
    setAttribute() {},
    focus() {}
  });

  const document = {
    getElementById: el,
    querySelector: () => el("q"),
    querySelectorAll: () => [],
    createElement: () => el("c" + Math.random()),
    addEventListener() {},
    body: el("body")
  };

  const isPathMNIST = fileName.startsWith("pathmnist_");
  const classId = isPathMNIST ? Number(fileName.split("_")[1]) : null;

  // Backend-ийн шинэ response хэлбэр (legacy + шинэ талбарууд)
  const fetch = async (url) => {
    calls.push(url);
    buttonDuringFetch = buttonDuringFetch || {
      disabled: els["ai-predict-button"].disabled,
      html: els["ai-predict-button"].innerHTML
    };
    if (mode === "down") throw new Error("server down");
    const body = url === "/predict"
      ? { label: "Микроскопийн зураг", class: "micro", confidence: 85, notice: "n",
          decision: "classified", decision_message: null, explanation: "e",
          dataset: isPathMNIST ? "PathMNIST" : "ToxoPath-1", analysis_id: 1 }
      : { class: classId, label: "L", confidence: 59.37, notice: "p",
          decision: "low_confidence", decision_message: "m", explanation: "e",
          morphology: "m", dataset: "PathMNIST", analysis_id: 2 };
    return { ok: true, json: async () => body };
  };

  new Function("document", "fetch", "FormData", "DataTransfer", "URL", "window", "alert", "Event", code)(
    document, fetch,
    class { append() {} },
    class { constructor() { this.items = { add() {} }; } },
    { createObjectURL: () => "" },
    { addEventListener() {}, location: { hash: "" }, scrollTo() {} },
    () => {},
    class {}
  );

  el("ai-image").files = [{ name: fileName }];
  // Бодит DOM шиг бүртгэгдсэн бүх listener-ийг дарааллаар нь дуудна
  const fire = async (key) => { for (const fn of handlers[key] || []) await fn(); };
  await fire("ai-predict-buttonclick");

  const E = (id) => els[id];
  const pm = E("pathmnist-result");
  const toxo = E("pathology-learning");
  const macro = E("related-macro");
  const label = fileName + (mode !== "ok" ? " [" + mode + "]" : "");
  console.log(`${label.padEnd(34)} calls=${calls.join("+").padEnd(28)} pathmnist=${pm.style.display} toxopath=${toxo.style.display} macro=${macro.style.display || "-"}`);

  check(buttonDuringFetch.disabled && buttonDuringFetch.html.includes("ai-spinner"), label + " loading state");
  check(!E("ai-predict-button").disabled && E("ai-predict-button").innerHTML.includes("AI-аар шинжлэх"), label + " button restored");
  check(E("ai-reset-button").style.display === "inline-flex", label + " reset shown");

  if (mode === "down") {
    check(E("ai-result").textContent.includes("алдаа"), label + " error text");
    check(scrolls.length === 0, label + " no scroll on error");
    return;
  }

  check(calls.length === (isPathMNIST ? 2 : 1), label + " request count");
  check(calls.includes("/predict-pathology") === isPathMNIST, label + " pathology endpoint only for PathMNIST");
  check(pm.style.display === (isPathMNIST ? "block" : "none"), label + " PathMNIST card visibility");
  check(toxo.style.display === (isPathMNIST ? "none" : "block"), label + " ToxoPath card visibility");
  check(E("results-dataset-badge").textContent === (isPathMNIST ? "PathMNIST" : "ToxoPath-1"), label + " dataset badge");
  check(E("results-header").hidden === false, label + " results header shown");
  check(E("ai-result").innerHTML.includes(">MICRO<") && E("ai-result").innerHTML.includes("width: 85%"), label + " image type card + bar");
  await fire("ai-imagechange");
  check(E("ai-file-name").textContent === fileName, label + " file name shown on change");
  if (isPathMNIST) {
    check(pm.innerHTML.includes("Морфологийн онцлог") && pm.innerHTML.includes("width: 59.37%"), label + " PathMNIST learning + bar");
    check(pm.innerHTML.includes("LOW CONFIDENCE") && pm.innerHTML.includes("callout-warning"), label + " low_confidence badge + callout");
  }
  if (/cardiac|myocarditis/.test(fileName)) {
    check(macro.style.display === "block" && macro.innerHTML.includes("Heart_001"), label + " heart macro");
  }

  await fire("ai-reset-buttonclick");
  check(["ai-result", "pathmnist-result", "pathology-learning", "related-macro", "ai-disclaimer", "ai-reset-button"]
    .every((id) => E(id).style.display === "none"), label + " reset hides all");
  check(E("ai-image").value === "" && E("ai-preview-container").style.display === "none", label + " reset input");
  check(E("results-header").hidden === true && E("ai-file-name").textContent === "", label + " reset header + file name");
}

(async () => {
  for (const name of [
    "pathmnist_2_debris.png",
    "pathmnist_8_adenocarcinoma.png",
    "cardiac_bradyzoite.png",
    "myocarditis_high.png",
    "brain_001.png"
  ]) {
    await run(name);
  }
  await run("cardiac_bradyzoite.png", "down");

  console.log(`\nfrontend checks passed: ${passed}, failed: ${failed}`);
  process.exit(failed === 0 ? 0 : 1);
})();
