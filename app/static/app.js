const STATUS = {
  downloading: "Téléchargement",
  diarizing: "Analyse des voix",
  ready: "Prêt",
  extracting: "Extraction",
  transcribing: "Transcription",
  done: "Terminé",
  error: "Erreur",
};

const ACTIVE = new Set(["downloading", "diarizing", "extracting", "transcribing"]);

const form = document.querySelector("#job-form");
const urlInput = document.querySelector("#url");
const fileInput = document.querySelector("#file");
const speakersInput = document.querySelector("#num-speakers");
const submitButton = document.querySelector("#submit");
const formError = document.querySelector("#form-error");
const banner = document.querySelector("#banner");
const statusBox = document.querySelector("#status");
const statusLabel = document.querySelector("#status-label");
const statusTitle = document.querySelector("#status-title");
const statusMessage = document.querySelector("#status-message");
const statusError = document.querySelector("#status-error");
const pulse = document.querySelector("#pulse");
const deleteButton = document.querySelector("#delete-job");
const voices = document.querySelector("#voices");
const voiceList = document.querySelector("#voice-list");
const result = document.querySelector("#result");
const output = document.querySelector("#output");
const downloadAudio = document.querySelector("#download-audio");
const transcribeButton = document.querySelector("#transcribe");
const downloadText = document.querySelector("#download-text");
const language = document.querySelector("#language");
const transcript = document.querySelector("#transcript");

let jobId = null;
let pollTimer = null;
let speakerKey = "";

form.addEventListener("submit", onSubmit);
deleteButton.addEventListener("click", removeJob);
transcribeButton.addEventListener("click", transcribe);
checkHealth();

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const notes = [];
    if (!data.ffmpeg) {
      notes.push("ffmpeg est introuvable. Installez-le, puis relancez le programme.");
    }
    if (!data.hf_token_configured) {
      notes.push(
        "Ajoutez HF_TOKEN dans le fichier .env pour identifier les voix. Le jeton sert à télécharger le modèle, l’audio reste ici."
      );
    }
    banner.hidden = notes.length === 0;
    banner.textContent = notes.join(" ");
  } catch (_error) {
    banner.hidden = false;
    banner.textContent = "Le programme ne répond pas.";
  }
}

async function onSubmit(event) {
  event.preventDefault();
  hide(formError);
  const file = fileInput.files[0];
  const url = urlInput.value.trim();
  if (file && url) {
    show(formError, "Indiquez un lien ou un fichier, pas les deux.");
    return;
  }
  if (!file && !url) {
    show(formError, "Collez un lien YouTube ou choisissez un fichier audio.");
    return;
  }
  const body = new FormData();
  if (file) {
    body.append("file", file);
  } else {
    body.append("url", url);
  }
  if (speakersInput.value.trim()) {
    body.append("num_speakers", speakersInput.value.trim());
  }

  submitButton.disabled = true;
  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    if (!response.ok) {
      show(formError, await readError(response));
      return;
    }
    const data = await response.json();
    jobId = data.id;
    speakerKey = "";
    output.removeAttribute("src");
    delete output.dataset.token;
    poll();
  } catch (_error) {
    show(formError, "La requête a échoué.");
  } finally {
    submitButton.disabled = false;
  }
}

async function poll() {
  clearTimeout(pollTimer);
  if (!jobId) {
    return;
  }
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) {
      show(formError, await readError(response));
      return;
    }
    const job = await response.json();
    render(job);
    if (ACTIVE.has(job.status)) {
      pollTimer = setTimeout(poll, 1000);
    }
  } catch (_error) {
    pollTimer = setTimeout(poll, 2000);
  }
}

function render(job) {
  statusBox.hidden = false;
  statusLabel.textContent = STATUS[job.status] || job.status;
  statusTitle.textContent = job.title || "";
  pulse.hidden = !ACTIVE.has(job.status);
  if (job.status === "error") {
    statusMessage.textContent = "";
    show(statusError, job.error || job.message);
  } else {
    statusMessage.textContent = job.message || "";
    if (job.error) {
      show(statusError, job.error);
    } else {
      hide(statusError);
    }
  }

  const key = job.speakers.map((speaker) => speaker.id).join("|");
  if (job.speakers.length) {
    voices.hidden = false;
    if (key !== speakerKey) {
      speakerKey = key;
      drawSpeakers(job);
    } else {
      updateSpeakerState(job);
    }
  } else {
    voices.hidden = true;
    voiceList.replaceChildren();
    speakerKey = "";
  }

  result.hidden = !job.has_output;
  if (job.has_output) {
    const src = `/api/jobs/${job.id}/output?v=${job.output_revision || 0}`;
    if (output.dataset.token !== src) {
      output.src = src;
      output.dataset.token = src;
    }
    downloadAudio.href = `/api/jobs/${job.id}/output?dl=1&v=${job.output_revision || 0}`;
  }
  transcribeButton.disabled = !job.has_output || ACTIVE.has(job.status);
  const hasText = Boolean(job.transcript);
  transcript.hidden = !hasText;
  transcript.textContent = job.transcript || "";
  downloadText.hidden = !hasText;
  downloadText.href = `/api/jobs/${job.id}/transcript`;
  if (job.language_label) {
    language.hidden = false;
    language.textContent = `Langue détectée : ${job.language_label}`;
  } else {
    language.hidden = true;
  }
}

function drawSpeakers(job) {
  voiceList.replaceChildren();
  for (const speaker of job.speakers) {
    const card = document.createElement("article");
    card.className = "voice";
    card.dataset.id = speaker.id;

    const title = document.createElement("h3");
    title.textContent = speaker.label;
    const meta = document.createElement("p");
    meta.className = "hint";
    const prises = speaker.turn_count > 1 ? "prises de parole" : "prise de parole";
    meta.textContent = `${speaker.duration_label} · ${speaker.turn_count} ${prises}`;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = `/api/jobs/${job.id}/preview/${speaker.id}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "primary";
    button.textContent = "Extraire cette voix";
    button.addEventListener("click", () => extract([speaker.id]));

    card.append(title, meta, audio, button);
    voiceList.append(card);
  }
  updateSpeakerState(job);
}

function updateSpeakerState(job) {
  const busy = job.status === "extracting" || job.status === "transcribing";
  const preroll = document.querySelector("#preroll");
  if (preroll) {
    preroll.disabled = busy;
  }
  for (const card of voiceList.querySelectorAll(".voice")) {
    card.classList.toggle("selected", card.dataset.id === job.selected_speaker);
    const button = card.querySelector("button");
    button.disabled = busy;
  }
}

async function extract(speakerIds) {
  if (!jobId) {
    return;
  }
  hide(statusError);
  const response = await fetch(`/api/jobs/${jobId}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      speakers: speakerIds,
      preroll: document.querySelector("#preroll").checked,
    }),
  });
  if (!response.ok) {
    show(statusError, await readError(response));
    statusError.hidden = false;
    return;
  }
  render(await response.json());
  poll();
}

async function transcribe() {
  if (!jobId) {
    return;
  }
  const response = await fetch(`/api/jobs/${jobId}/transcribe`, { method: "POST" });
  if (!response.ok) {
    show(statusError, await readError(response));
    return;
  }
  render(await response.json());
  poll();
}

async function removeJob() {
  if (!jobId) {
    return;
  }
  const current = jobId;
  jobId = null;
  clearTimeout(pollTimer);
  await fetch(`/api/jobs/${current}`, { method: "DELETE" });
  speakerKey = "";
  statusBox.hidden = true;
  voices.hidden = true;
  result.hidden = true;
  voiceList.replaceChildren();
  output.removeAttribute("src");
  delete output.dataset.token;
  transcript.textContent = "";
}

async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") {
      return data.detail;
    }
  } catch (_error) {
    /* réponse non JSON */
  }
  return "La requête a échoué.";
}

function show(node, text) {
  node.hidden = false;
  node.textContent = text;
}

function hide(node) {
  node.hidden = true;
  node.textContent = "";
}
