// thumbsFeedback.js

/**
 * Envía un pulgar arriba/abajo al endpoint /feedback_rating
 */
export function sendThumbsFeedback({ evaluation, reason = "", containerEl }) {
  // === ADDED: Let's find the closest parent with class ".assistant-message"
  const parentMsg = containerEl.closest(".assistant-message");
  if (!parentMsg) {
    console.warn("No .assistant-message parent found. Cannot find runId!");
    return;
  }

  // === CHANGED: We read runId from parentMsg.dataset.runId
  const runId = parentMsg.dataset.runId || "";
  if (!runId) {
    console.warn("No run_id in parentMsg.dataset. Feedback not sent.");
    return;
  }

  const payload = {
    run_id: runId,
    evaluation: evaluation,   // "up" or "down"
    reason: reason            // reason is optional; only used for "down"
  };

  fetch("/thumb_feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(res => {
      if (!res.ok) {
        throw new Error("Error al enviar feedback");
      }
      return res.json();
    })
    .then(data => {
      console.log("Thumb feedback response:", data.message);

      // Show a small confirmation message in the UI
      const msgEl = document.createElement("div");
      msgEl.className = "thumbs-confirmation";
      msgEl.innerText = "¡Gracias por tu evaluación!";

      containerEl.appendChild(msgEl);

      // Optionally disable the thumbs buttons to prevent multiple submissions
      const buttons = containerEl.querySelectorAll(".thumbs-btn");
      buttons.forEach(btn => { btn.disabled = true; });
    })
    .catch(err => {
      console.error("Error sending thumbs feedback:", err);
      alert("No se pudo enviar el feedback. Intente de nuevo.");
    });
}

/**
 * Crea y agrega los botones de pulgar arriba/abajo debajo de la respuesta
 */
export function insertThumbsFeedbackUI({ assistantDiv }) {

  const feedbackContainer = document.createElement("div");
  feedbackContainer.className = "feedback-container";

  feedbackContainer.innerHTML = `
    <button class="thumbs-btn thumbs-up" data-type="up">
      <span class="thumb-icon"><i class="fas fa-thumbs-up"></i></span>
      <span class="thumb-label">Correcto</span>
    </button>
    <button class="thumbs-btn thumbs-down" data-type="down">
      <span class="thumb-icon"><i class="fas fa-thumbs-down"></i></span>
      <span class="thumb-label">Incorrecto</span>
    </button>
  `;

  assistantDiv.appendChild(feedbackContainer);

  const upBtn = feedbackContainer.querySelector('.thumbs-up');
  const downBtn = feedbackContainer.querySelector('.thumbs-down');

  upBtn.addEventListener("click", () => {
    sendThumbsFeedback({
      evaluation: "up",
      reason: "",
      containerEl: feedbackContainer
    });
  });

  downBtn.addEventListener("click", () => {
    showThumbsDownModal(feedbackContainer);
  });
}

/**
 * Called once DOM is ready
 * Finds the modal elements and wires up event listeners.
 */
export function initThumbsDownModal() {
  const thumbsDownModal = document.getElementById("thumbs-down-modal");
  const reasonTextarea = document.getElementById("thumbs-down-reason");
  const cancelBtn = document.getElementById("cancel-thumbs-down");
  const submitBtn = document.getElementById("submit-thumbs-down");

  if (!thumbsDownModal || !reasonTextarea || !cancelBtn || !submitBtn) {
    console.error("Thumbs-down modal elements not found in DOM.");
    return;
  }

  // Cancel => hide
  cancelBtn.addEventListener("click", () => {
    hideThumbsDownModal(thumbsDownModal);
  });

  // Submit => gather reason, send "down", hide modal
  submitBtn.addEventListener("click", () => {
    const reason = reasonTextarea.value.trim();

    if (reason === "") {
      alert("Por favor, proporciona una breve explicación.");
      return;
    }

    // Instead of question/answer, we only need evaluation, reason, container
    // We'll rely on our stored reference to containerEl
    sendThumbsFeedback({
      evaluation: "down",
      reason: reason,
      containerEl: currentContainerEl
    });

    hideThumbsDownModal(thumbsDownModal);
  });
}

/** Show the thumbs-down modal, storing the relevant data for submission. */
export function showThumbsDownModal(containerEl) {
  currentContainerEl = containerEl;

  const reasonTextarea = document.getElementById("thumbs-down-reason");
  if (reasonTextarea) {
    reasonTextarea.value = "";
  }

  const thumbsDownModal = document.getElementById("thumbs-down-modal");
  if (thumbsDownModal) {
    thumbsDownModal.style.display = "flex";
  }
}


/** Hide the thumbs-down modal. */
function hideThumbsDownModal(thumbsDownModal) {
  if (thumbsDownModal) {
    thumbsDownModal.style.display = "none";
  }
}

// We'll store only containerEl. (Removed currentQuestion/currentAnswer.)
let currentContainerEl = null;


