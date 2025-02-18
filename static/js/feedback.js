// feedback.js
export function initFeedback() {
  const feedbackModal = document.getElementById("feedback-modal");
  const feedbackButton = document.getElementById("feedback-button");
  const closeFeedbackButton = document.getElementById("close-feedback");
  const submitFeedbackButton = document.getElementById("submit-feedback");
  const feedbackTextArea = document.getElementById("feedback-text");

  if (feedbackButton) {
    feedbackButton.addEventListener("click", () => {
      feedbackModal.style.display = "block";
    });
  }

  function closeFeedbackModal() {
    feedbackModal.style.display = "none";
    // Optionally reset textarea:
    // feedbackTextArea.value = "";
  }

  if (closeFeedbackButton) {
    closeFeedbackButton.addEventListener("click", closeFeedbackModal);
  }

  if (submitFeedbackButton) {
    submitFeedbackButton.addEventListener("click", () => {
      const feedbackText = feedbackTextArea.value.trim();
      if (!feedbackText) {
        alert("Por favor, ingrese un comentario primero.");
        return;
      }
      fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback: feedbackText }),
      })
        .then(response => {
          if (!response.ok) {
            throw new Error("Error al enviar feedback");
          }
          return response.json();
        })
        .then(data => {
          alert(data.message || "¡Gracias por tu feedback!");
          closeFeedbackModal();
          feedbackTextArea.value = "";
        })
        .catch(err => {
          console.error("Error:", err);
          alert("No se pudo enviar el feedback. Intente de nuevo.");
        });
    });
  }
}
