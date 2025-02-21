// chatUI.js

import { initOsmaSession } from './osmaHandler.js';
import { sendMessageStream, abortController } from './streamHandler.js';

/**
 * Initializes the chat UI by appending a welcome message.
 */
export function initChatUI() {
  const chatBox = document.getElementById("chat-box");
  const welcomeMessage = document.createElement("div");
  welcomeMessage.className = "assistant-message";
  const domain = window.DOMAIN || "Dominio de asistencia";
  const company = window.COMPANY || "Mi Empresa";
   welcomeMessage.innerText = `Hola, soy un asistente especializado en ${domain} de ${company}, ¿En qué puedo ayudarte?`;
  chatBox.appendChild(welcomeMessage);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Fetch recommended questions from JSON and dynamically create the .options-wrapper block
 */
export async function loadRecommendedQuestions() {
  try {
  const company = window.COMPANY || "default_company";
  const response = await fetch(`/static/${window.COMPANY}/data/options-wrapper.json`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    const questions = data.questions || [];

    // Create the main wrapper
    const optionsWrapper = document.createElement('div');
    optionsWrapper.className = 'options-wrapper';

    // Create the container for your recommendation boxes
    const coffeeOptionsContainer = document.createElement('div');
    coffeeOptionsContainer.className = 'options-container coffee-options';

    // Fill with question boxes
    questions.forEach(question => {
      const optionBox = document.createElement('div');
      optionBox.className = 'option-box';
      optionBox.textContent = question;
      coffeeOptionsContainer.appendChild(optionBox);
    });

    // If OSMA is enabled, optionally add that container
    if (window.OSMA_ENABLED) {
      const osmaContainer = document.createElement('div');
      osmaContainer.className = 'options-container osma-option';

      const osmaBox = document.createElement('div');
      osmaBox.className = 'option-box osma';
      osmaBox.textContent = 'Exploración de datos con OSMA';
      osmaContainer.appendChild(osmaBox);

      // Append OSMA container to the wrapper
      optionsWrapper.appendChild(osmaContainer);
    }

    // Append coffeeOptionsContainer to the wrapper
    optionsWrapper.appendChild(coffeeOptionsContainer);

    // Finally, append everything to the chat box
    const chatBox = document.getElementById("chat-box");
    if (chatBox) {
      chatBox.appendChild(optionsWrapper);
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  } catch (err) {
    console.error("Error fetching the recommended questions JSON:", err);
  }
}

/**
 * Appends a user message to the chat box.
 */
export function appendUserMessage(message) {
    console.log("appendUserMessage called with message:", message); // Added log
  // Eliminar el cuadro de cambio de OSMA si existe
  const osmaModeBox = document.querySelector('.osma-mode-switch');
  if (osmaModeBox) {
    console.log("Removing existing OSMA mode switch box."); // Added log
    osmaModeBox.remove();
  }

  const chatBox = document.getElementById("chat-box");
  const userMessage = document.createElement("div");
  userMessage.className = "user-message";
  userMessage.innerText = message;
  chatBox.appendChild(userMessage);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Appends an assistant message container to the chat box.
 * Returns the created element.
 */
export function appendAssistantMessage(htmlText = "") {
  console.log("appendAssistantMessage called with htmlText:", htmlText); // Added log
  const chatBox = document.getElementById("chat-box");
  const assistantMessage = document.createElement("div");
  assistantMessage.className = "assistant-message";
  assistantMessage.innerHTML = htmlText;
  chatBox.appendChild(assistantMessage);
  chatBox.scrollTop = chatBox.scrollHeight;
  return assistantMessage;
}

/**
 * Returns a new typing indicator element.
 */
export function createTypingIndicator() {
  const indicator = document.createElement("div");
  indicator.className = "assistant-message";
  indicator.innerHTML =
    '<span class="typing-indicator"></span><span class="typing-indicator"></span><span class="typing-indicator"></span>';
  return indicator;
}

/**
 * A simple typewriter effect that gradually adds text to an element.
 */
export function typeWriterEffect(element, text) {
  let index = 0;
  function type() {
    if (index < text.length) {
      element.innerHTML += text.charAt(index);
      index++;
      setTimeout(type, 50);
    } else {
      if (window.MathJax) {
        window.MathJax.typesetPromise([element]).catch(err => console.error("MathJax typeset error:", err));
      }
    }
  }
  type();
}



/**
 * Borrar último mensaje (usuario + asistente), abort streaming, y notificar backend
 */
export function eraseLastAndStop() {
  // 1) Remove the last user message from DOM
  removeLastUserMessage();

  // 2) Remove the last assistant message from DOM
  removeLastAssistantMessage();

  // 3) Abort streaming if active
  if (abortController) {
    console.log("Aborting streaming.");
    abortController.abort();
    // No need to set abortController to null; streamHandler.js handles it
  }
}

window.eraseLastAndStop = eraseLastAndStop;

/**
 * Agrega al final del chat un cuadro para cambiar a "Exploración de datos con OSMA".
 * Se asume que al hacer clic se dispara sendOptionMessage() con el mensaje correspondiente.
 */
export function appendOsmaModeSwitchBox() {
    if (!window.OSMA_ENABLED) {
    console.log("OSMA mode is disabled; appendOsmaModeSwitchBox() will do nothing.");
    return;
  }
  console.log("appendOsmaModeSwitchBox called."); // Added log
  // Elimina cualquier cuadro existente para evitar duplicados.
  const existingBox = document.querySelector('.osma-mode-switch');
  if (existingBox) {
    console.log("Existing OSMA mode switch box found. Removing it."); // Added log
    existingBox.remove();
  }

  const chatBox = document.getElementById("chat-box");
  const osmaBox = document.createElement("div");
  osmaBox.className = "osma-mode-switch";
  osmaBox.innerText = "Importar datos de OSMA";

  // Al hacer clic, se inicia la sesión OSMA.
  osmaBox.addEventListener("click", function() {
    console.log("OSMA mode switch box clicked."); // Added log
    // Llama directamente a la función que inicia la sesión OSMA
    initOsmaSession();
    // Activa el modo OSMA para que desde ese momento se envíen las respuestas a OSMA
    window.isOSMASession = true;
    // Se elimina el cuadro para evitar duplicados
    osmaBox.remove();
  });

  chatBox.appendChild(osmaBox);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Removes the last user message from the chat box.
 */
function removeLastUserMessage() {
  console.log("removeLastUserMessage called."); // Added log
  const chatBox = document.getElementById("chat-box");
  const userMessages = chatBox.querySelectorAll(".user-message");
  if (userMessages.length > 0) {
    userMessages[userMessages.length - 1].remove();
  }
}

/**
 * Removes the last assistant message from the chat box.
 */
function removeLastAssistantMessage() {
  console.log("removeLastAssistantMessage called."); // Added log
  const chatBox = document.getElementById("chat-box");
  const assistantMessages = chatBox.querySelectorAll(".assistant-message");
  if (assistantMessages.length > 0) {
    assistantMessages[assistantMessages.length - 1].remove();
  }
}

/**
 * Hides Top part of the page after first message is sent
 */

export function hideOptionContainers() {
  console.log("hideOptionContainers called.");

  const welcomeMessageEl = document.querySelector(".welcome-message");
  const coffeeOptionsContainerEl = document.querySelector(".options-container.coffee-options");
  const osmaOptionsContainerEl = document.querySelector(".options-container.osma-option");

  // NEW: also hide the .options-wrapper
  const optionsWrapperEl = document.querySelector(".options-wrapper");

  if (welcomeMessageEl) {
    welcomeMessageEl.style.display = "none";
  }
  if (coffeeOptionsContainerEl) {
    coffeeOptionsContainerEl.style.display = "none";
  }
  if (osmaOptionsContainerEl) {
    osmaOptionsContainerEl.style.display = "none";
  }

  // Hide the parent wrapper
  if (optionsWrapperEl) {
    optionsWrapperEl.style.display = "none";
  }

  console.log("Ocultando .welcome-message, .coffee-options, .osma-option, y .options-wrapper.");
}

/**
 * Scrolls the chat-box to the bottom smoothly.
 */
export function scrollToBottom() {
  const chatBox = document.getElementById("chat-box");
  if (chatBox) {
    chatBox.scrollTo({
      top: chatBox.scrollHeight,
      behavior: "smooth" // Use "auto" for instant scroll
    });
  }
}




