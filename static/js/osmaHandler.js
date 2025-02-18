// osmaHandler.js

import { appendAssistantMessage, appendOptionContainers, hideOptionContainers } from './chatUI.js';

/* Estado global del flujo OSMA (puedes usar window.osmaState que se actualiza en el backend,
   o mantener uno local en el frontend; aquí usamos uno propio para la interacción visual) */
window.osmaFlowState = {
  step: 0,
  services: [],       // servicios seleccionados
  monitoreables: [],  // monitoreables seleccionados
  variables: [],      // variables seleccionadas
  fechaInicio: null,
  fechaFin: null,
  intervalo: null
};


/**
 * Función para iniciar la sesión OSMA.
 * Llama al endpoint /osma_init y muestra el prompt inicial y las opciones de servicios.
 */
export function initOsmaSession() {
  if (!window.OSMA_ENABLED) {
    console.log("OSMA mode is disabled; initOsmaSession() will do nothing.");
    return;
  }
  console.log("initOsmaSession called."); // Added log
  fetch("/osma_init", {
    method: "POST"
  })
    .then(response => response.json())
    .then(data => {
      if (data.prompt && data.services) {
        appendAssistantMessage(`<em>${data.prompt}</em>`);
        // Aquí puedes crear dinámicamente un formulario basado en data.services
        showServiceForm(data.services);
        window.isOSMASession = true;
      } else if (data.error) {
        appendAssistantMessage(`<em>Error: ${data.error}</em>`);
      }
    })
    .catch(err => {
      console.error("Error iniciando sesión OSMA:", err);
    });
}

/**
 * Muestra un formulario de selección de servicios.
 * @param {Array} servicesAvailable - arreglo de servicios obtenido del backend.
 */
function showServiceForm(servicesAvailable) {
  if (!window.OSMA_ENABLED) return;
  hideOptionContainers(); // Oculta los contenedores de opciones existentes
  const chatBox = document.getElementById("chat-box");const formHTML = `
  <div class="osma-form" id="osma-service-form">
    <h4>Seleccione los servicio(s):</h4>
    <div class="checkbox-list">
      ${servicesAvailable
        .map(service => `<label><input type="checkbox" value="${service}" /> ${service}</label>`)
        .join("")}
    </div>
    <button id="osma-service-next">Siguiente</button>
  </div>
`;
  chatBox.innerHTML += formHTML;
  document.getElementById("osma-service-next").addEventListener("click", () => {
    const selected = Array.from(
      document.querySelectorAll("#osma-service-form input[type=checkbox]:checked")
    ).map(input => input.value);
    if (selected.length === 0) {
      alert("Seleccione al menos un servicio.");
      return;
    }
    window.osmaFlowState.services = selected;
    window.osmaFlowState.step = 1; // Increment step
    sendOsmaAnswer(selected.join(","));
  });
}

/**
 * Muestra el formulario de Monitoreables. Los datos se solicitan al backend.
 * La respuesta del endpoint /osma_respond incluirá la lista de monitoreables.
 */
function showMonitoreablesForm(monitoreablesAvailable) {
  clearOsmaForms();
  const chatBox = document.getElementById("chat-box");
  const formHTML = `
    <div class="osma-form" id="osma-monitoreables-form">
      <h4>Seleccione los monitoreables:</h4>
      ${monitoreablesAvailable
        .map(mon => `<label><input type="checkbox" value="${mon}" /> ${mon}</label>`)
        .join("<br>")}
      <br>
      <button id="osma-monitoreables-next">Siguiente</button>
    </div>
  `;
  chatBox.innerHTML += formHTML;
  document.getElementById("osma-monitoreables-next").addEventListener("click", () => {
    const selected = Array.from(
      document.querySelectorAll("#osma-monitoreables-form input[type=checkbox]:checked")
    ).map(input => input.value);
    if (selected.length === 0) {
      alert("Seleccione al menos un monitoreable.");
      return;
    }
    window.osmaFlowState.monitoreables = selected;
    window.osmaFlowState.step = 2; // Increment step
    sendOsmaAnswer(selected.join(","));
  });
}

/**
 * Muestra el formulario de Variables. Los datos se pasan del backend.
 */
function showVariablesForm(variablesAvailable) {
  clearOsmaForms();
  const chatBox = document.getElementById("chat-box");
  const formHTML = `
    <div class="osma-form" id="osma-variables-form">
      <h4>Seleccione las variables:</h4>
      ${variablesAvailable
        .map(v => `<label><input type="checkbox" value="${v}" /> ${v}</label>`)
        .join("<br>")}
      <br>
      <button id="osma-variables-next">Siguiente</button>
    </div>
  `;
  chatBox.innerHTML += formHTML;
  document.getElementById("osma-variables-next").addEventListener("click", () => {
    const selected = Array.from(
      document.querySelectorAll("#osma-variables-form input[type=checkbox]:checked")
    ).map(input => input.value);
    if (selected.length === 0) {
      alert("Seleccione al menos una variable.");
      return;
    }
    window.osmaFlowState.variables = selected;
    window.osmaFlowState.step = 3; // Increment step
    sendOsmaAnswer(selected.join(","));
  });
}

/**
 * Muestra un formulario para ingresar la fecha/hora de inicio y fin con límite de 1 mes.
 */
function showDateForm() {
  clearOsmaForms();
  const chatBox = document.getElementById("chat-box");
  const formHTML = `
    <div class="osma-form" id="osma-date-form">
      <h4>Indique la fecha y hora de inicio y fin (máx. 1 mes):</h4>
      <label>Inicio: <input type="datetime-local" id="osma-date-start" /></label><br>
      <label>Fin: <input type="datetime-local" id="osma-date-end" /></label><br>
      <button id="osma-date-next">Siguiente</button>
    </div>
  `;
  chatBox.innerHTML += formHTML;
  document.getElementById("osma-date-next").addEventListener("click", () => {
    const start = document.getElementById("osma-date-start").value;
    const end = document.getElementById("osma-date-end").value;
    if (!start || !end) {
      alert("Debe completar ambas fechas.");
      return;
    }
    const diffDays = (new Date(end) - new Date(start)) / (1000 * 60 * 60 * 24);
    if (diffDays > 31) {
      alert("El rango máximo permitido es de 31 días (1 mes).");
      return;
    }
    window.osmaFlowState.fechaInicio = start;
    window.osmaFlowState.fechaFin = end;
    window.osmaFlowState.step = 4; // Increment step
    sendOsmaAnswer(`${start},${end}`);
  });
}

/**
 * Muestra el formulario para seleccionar el intervalo de consulta.
 */
function showIntervalForm(intervalsAvailable) {
  clearOsmaForms();
  const chatBox = document.getElementById("chat-box");
  const formHTML = `
    <div class="osma-form" id="osma-interval-form">
      <h4>Seleccione el intervalo de consulta:</h4>
      ${intervalsAvailable
        .map(i => `<label><input type="radio" name="osma-interval" value="${i}" /> ${i}</label>`)
        .join("<br>")}
      <br>
      <button id="osma-interval-next">Siguiente</button>
    </div>
  `;
  chatBox.innerHTML += formHTML;
  document.getElementById("osma-interval-next").addEventListener("click", () => {
    const selected = document.querySelector('input[name="osma-interval"]:checked');
    if (!selected) {
      alert("Seleccione un intervalo.");
      return;
    }
    window.osmaFlowState.intervalo = selected.value;
    window.osmaFlowState.step = 5; // Increment step
    sendOsmaAnswer(selected.value);
  });
}

/**
 * Muestra el resumen final de la consulta y un botón para confirmarla.
 */
function showFinalConfirmation(finalPrompt) {
  clearOsmaForms();
  const { services, monitoreables, variables, fechaInicio, fechaFin, intervalo } = window.osmaFlowState;
  const chatBox = document.getElementById("chat-box");
  const summaryHTML = `
    <div class="osma-form" id="osma-summary">
      <h4>Resumen de la consulta OSMA</h4>
      <p><strong>Servicios:</strong> ${services.join(", ")}</p>
      <p><strong>Monitoreables:</strong> ${monitoreables.join(", ")}</p>
      <p><strong>Variables:</strong> ${variables.join(", ")}</p>
      <p><strong>Fechas:</strong> ${fechaInicio} a ${fechaFin}</p>
      <p><strong>Intervalo:</strong> ${intervalo}</p>
      <button id="osma-confirm">Ejecutar consulta OSMA</button>
    </div>
  `;
  chatBox.innerHTML += summaryHTML;
  document.getElementById("osma-confirm").addEventListener("click", () => {
    // Aquí se llama al backend para ejecutar la consulta final.
    executeOsmaQuery({ ...window.osmaFlowState });
  });
}


/**
 * Envía la respuesta del usuario al endpoint /osma_respond para actualizar el flujo OSMA.
 * La respuesta enviada es el valor (o valores separados por comas) del formulario actual.
 */
function sendOsmaAnswer(answer) {
  fetch("/osma_respond", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ respuesta: answer })
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        appendAssistantMessage(`<em>Error: ${data.error}</em>`);
        return;
      }

      const nextPrompt = data.prompt;
      appendAssistantMessage(`<em>${nextPrompt}</em>`);

      if (data.monitoreables) {
        showMonitoreablesForm(data.monitoreables);
      } else if (data.variables) {
        showVariablesForm(data.variables);
      } else if (data.intervals) {
        showIntervalForm(data.intervals);
      } else if (window.osmaFlowState.step === 3) { // Handle date form
        showDateForm();
      } else if (window.osmaFlowState.step === 5) { // Final confirmation
        showFinalConfirmation(nextPrompt);
      }
    })
    .catch(err => {
      console.error("Error en el flujo OSMA:", err);
    });
}

/**
 * Ejecuta la consulta OSMA con la información acumulada y reinicia el flujo.
 * Aquí puedes hacer una llamada fetch final al backend que realice la consulta.
 */
function executeOsmaQuery(queryData) {
  // Ejemplo: se envía al endpoint /osma_query (debes implementarlo en el backend)
  fetch("/osma_query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(queryData)
  })
    .then(response => response.json())
    .then(data => {
      appendAssistantMessage(`<em>Consulta ejecutada. Respuesta: ${data.resultado}</em>`);
      // Reinicia el modo OSMA
      window.isOSMASession = false;
      window.osmaFlowState = { step: 0, services: [], monitoreables: [], variables: [], fechaInicio: null, fechaFin: null, intervalo: null };
      // Aquí puedes agregar una opción para volver al flujo normal o para reiniciar OSMA.
    })
    .catch(err => {
      console.error("Error ejecutando la consulta OSMA:", err);
    });
}

/**
 * Elimina cualquier formulario OSMA que esté presente.
 */
function clearOsmaForms() {
  if (!window.OSMA_ENABLED) return;
  document.querySelectorAll(".osma-form").forEach(el => el.remove());
}

/**
 * Solicita al usuario confirmar si desea abortar el proceso OSMA.
 * Se muestra un pequeño formulario con botones de “Sí” y “No”.
 */
export function promptAbortProcess() {
  if (!window.OSMA_ENABLED) return;
  console.log("[promptAbortProcess] Se invoca el prompt de aborto.");

  // If an existing .osma-abort is in DOM, remove it first
  const oldAbortBox = document.getElementById("osma-abort");
  if (oldAbortBox) oldAbortBox.remove();

  const chatBox = document.getElementById("chat-box");
  const abortHTML = `
    <div class="osma-form" id="osma-abort">
      <h4>¿Quiere abortar el proceso de importar datos de OSMA?</h4>
      <button id="osma-abort-yes">Sí</button>
      <button id="osma-abort-no">No</button>
    </div>
  `;
  chatBox.innerHTML += abortHTML;

  // Attach Event Listeners
  const abortYesButton = document.getElementById("osma-abort-yes");
  const abortNoButton = document.getElementById("osma-abort-no");

  if (abortYesButton) {
    abortYesButton.addEventListener("click", () => {
      console.log("[promptAbortProcess] Usuario confirmó ABORTAR OSMA.");
      window.isOSMASession = false;
      window.osmaFlowState = { step: 0, services: [], monitoreables: [], variables: [], fechaInicio: null, fechaFin: null, intervalo: null };
      document.getElementById("osma-abort").remove();
      clearOsmaForms();
      appendAssistantMessage(`<em>Proceso OSMA abortado. Regresando al flujo normal.</em>`);

      // Append the option containers dynamically
      appendOptionContainers();
    });
  }

  if (abortNoButton) {
    abortNoButton.addEventListener("click", () => {
      console.log("[promptAbortProcess] Usuario decidió CONTINUAR OSMA.");
      document.getElementById("osma-abort").remove();
    });
  }
}

