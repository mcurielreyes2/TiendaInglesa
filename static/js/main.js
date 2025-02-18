// main.js

import { initChatUI, appendUserMessage, appendOsmaModeSwitchBox, hideOptionContainers, appendOptionContainers } from './chatUI.js';
import { sendMessageStream, sendOptionMessage } from './streamHandler.js';
import { initFeedback } from './feedback.js';
import { initOsmaSession, promptAbortProcess } from './osmaHandler.js';
import { initThumbsDownModal } from './thumbsFeedback.js';

window.OSMA_ENABLED = false; // or false

// Establish a global flag for OSMA session
window.isOSMASession = false;

window.addEventListener("DOMContentLoaded", () => {
  console.log("DOMContentLoaded event triggered."); // Added log
  // Initialize UI
  initChatUI();

  // Initialize feedback
  initFeedback();

  // Initialize the thumbs-down modal
  initThumbsDownModal();

  // Get input and send button
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");

  // Add listener to send button
  if (sendButton) {
    sendButton.addEventListener("click", () => {
      const message = userInput.value.trim();
      if (message !== "") {
        sendOptionMessage(message);
        // Clear the input after sending
        userInput.value = "";

      }
    });
  }

userInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    console.log("Enter key pressed in user-input."); // Log to track keypress
    sendMessageStream(); // Ensure sendMessageStream handles message retrieval
    userInput.value = "";
  }
});

  // Event Delegation for Static Option Boxes in `.options-wrapper`
  const optionsWrapper = document.querySelector('.options-wrapper');
  if (optionsWrapper) {
    optionsWrapper.addEventListener('click', (event) => {
      const box = event.target.closest('.option-box');
      if (box) {
        console.log("Option box clicked inside .options-wrapper:", box.innerText.trim()); // Added log
        event.stopPropagation();
        event.preventDefault();
        const isOsma = box.classList.contains('osma');

        // Removed appendUserMessage call
        hideOptionContainers();

        if (isOsma) {
          initOsmaSession();
          window.isOSMASession = true;
          console.log("OSMA mode activated from option-box OSMA.");
        } else {
          sendOptionMessage(box.innerText.trim());
        }
      }
    });
  }

// Event Delegation for Dynamically Added Option Boxes in `#chat-box`
const chatBox = document.getElementById("chat-box");
if (chatBox) {
  chatBox.addEventListener('click', (event) => {
    const box = event.target.closest('.option-box');
    if (box) {
      console.log("Option box clicked inside #chat-box:", box.innerText.trim()); // Added log
      event.stopPropagation();
      event.preventDefault();
      const isOsma = box.classList.contains('osma');

      // Removed appendUserMessage call
      hideOptionContainers();

      if (isOsma) {
        initOsmaSession();
        window.isOSMASession = true;
        console.log("OSMA mode activated from option-box OSMA.");
      } else {
        sendOptionMessage(box.innerText.trim());
      }
    }
  });
}

// OSMA button (if exists)
const osmaButton = document.getElementById("osma-button");
if (osmaButton) {
  if (window.OSMA_ENABLED) {
    // OSMA is enabled: set up the click event
    osmaButton.addEventListener("click", () => {
      console.log("OSMA button clicked."); // Added log
      appendUserMessage(osmaButton.innerText.trim());
      hideOptionContainers();
      initOsmaSession();
      window.isOSMASession = true;
      console.log("Modo OSMA activado desde el botón OSMA.");
      osmaButton.style.display = "none";
    });
  } else {
    // OSMA is disabled: just hide or remove the button
    osmaButton.style.display = "none";
  }
}


  // Expose sendMessageStream globally
  window.sendMessageStream = sendMessageStream;
});