const root = document.documentElement;
const board = document.querySelector("#board-scroll");
const dialog = document.querySelector("#plot-dialog");
const dialogImage = document.querySelector("#dialog-image");
const dialogTitle = document.querySelector("#dialog-title");

const widths = [460, 540, 620, 700, 780];
let widthIndex = 2;

function applyWidth() {
  root.style.setProperty("--column-width", `${widths[widthIndex]}px`);
}

document.querySelector("#zoom-out").addEventListener("click", () => {
  widthIndex = Math.max(0, widthIndex - 1);
  applyWidth();
});

document.querySelector("#zoom-in").addEventListener("click", () => {
  widthIndex = Math.min(widths.length - 1, widthIndex + 1);
  applyWidth();
});

document.querySelector("#zoom-reset").addEventListener("click", () => {
  widthIndex = 2;
  applyWidth();
});

let dragState = null;

board.addEventListener("pointerdown", (event) => {
  if (event.target.closest("button")) {
    return;
  }
  dragState = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    scrollLeft: board.scrollLeft,
    scrollTop: board.scrollTop
  };
  board.setPointerCapture(event.pointerId);
  board.classList.add("dragging");
});

board.addEventListener("pointermove", (event) => {
  if (!dragState || dragState.pointerId !== event.pointerId) {
    return;
  }
  board.scrollLeft = dragState.scrollLeft - (event.clientX - dragState.x);
  board.scrollTop = dragState.scrollTop - (event.clientY - dragState.y);
});

function stopDragging(event) {
  if (!dragState || dragState.pointerId !== event.pointerId) {
    return;
  }
  dragState = null;
  board.classList.remove("dragging");
}

board.addEventListener("pointerup", stopDragging);
board.addEventListener("pointercancel", stopDragging);

document.querySelectorAll(".expand-button").forEach((button) => {
  button.addEventListener("click", () => {
    dialogImage.src = button.dataset.image;
    dialogImage.alt = `${button.dataset.title} full-resolution plot`;
    dialogTitle.textContent = button.dataset.title;
    dialog.showModal();
  });
});

document.querySelector("#dialog-close").addEventListener("click", () => {
  dialog.close();
});

dialog.addEventListener("click", (event) => {
  if (event.target === dialog) {
    dialog.close();
  }
});
