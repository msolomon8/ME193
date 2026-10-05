// Streams this computer's webcam to the UNO Q and draws what the board detects.
//
// Flow: grab a frame -> send it to the board as JPEG ("frame") -> the board runs both
// models and answers with "detections" -> draw the boxes -> send the next frame.
// Waiting for the answer keeps frames from piling up when the board is slow.

const WIDTH = 640, HEIGHT = 480;
const GREEN_TARGET = 0.25, BLUE_TARGET = 0.75;   // must match the cars' TARGET_X
const RESEND_MS = 2000;                          // resend if no answer comes back (e.g. lost frame)

const video = document.querySelector('#cam');
const view = document.querySelector('#view');
const ctx = view.getContext('2d');
const startBtn = document.querySelector('#start-btn');
const statusEl = document.querySelector('#status');
const readout = document.querySelector('#readout');

// Arduino's libs/arduino.js always connects over http://, but this page is https://
// (needed for the webcam), so connect to the same origin with socket.io directly.
const socket = io(window.location.origin);

const grab = document.createElement('canvas');   // offscreen canvas for the JPEG we send
grab.width = WIDTH;
grab.height = HEIGHT;
const grabCtx = grab.getContext('2d');

let last = { green: null, blue: null, ms: 0 };
let waiting = false;
let resendTimer = null;

socket.on('connect', () => {
  statusEl.textContent = 'Connected to UNO Q';
  if (startBtn.disabled) sendFrame();    // camera already running -> resume after a reconnect
});
socket.on('disconnect', () => { statusEl.textContent = 'Disconnected'; waiting = false; });

socket.on('detections', (d) => {
  last = d;
  waiting = false;
  clearTimeout(resendTimer);
  sendFrame();
});

startBtn.addEventListener('click', async () => {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: WIDTH, height: HEIGHT }, audio: false,
    });
  } catch (e) {
    statusEl.textContent = `Camera error: ${e.message}`;
    return;
  }
  startBtn.disabled = true;
  requestAnimationFrame(draw);
  sendFrame();                           // retries until the camera and socket are ready
});

function sendFrame() {
  if (waiting) return;                   // the answer (or the resend timer) calls us again
  if (!socket.connected || video.readyState < 2) {
    setTimeout(sendFrame, 200);
    return;
  }
  waiting = true;
  grabCtx.drawImage(video, 0, 0, WIDTH, HEIGHT);
  grab.toBlob(async (blob) => {
    socket.emit('frame', await blob.arrayBuffer());
    resendTimer = setTimeout(() => { waiting = false; sendFrame(); }, RESEND_MS);
  }, 'image/jpeg', 0.7);
}

function draw() {
  ctx.drawImage(video, 0, 0, WIDTH, HEIGHT);

  // Stopping lines
  line(GREEN_TARGET, '#0c0');
  line(BLUE_TARGET, '#05f');

  // Latest boxes from the board (normalized 0-1)
  box(last.green, '#0c0', 'Green Minifig');
  box(last.blue, '#05f', 'Blue Minifig');

  const fmt = (b) => (b ? `x=${b.x.toFixed(2)}` : 'not seen');
  readout.textContent =
    `green: ${fmt(last.green)}   blue: ${fmt(last.blue)}   board: ${last.ms} ms/frame`;
  requestAnimationFrame(draw);
}

function line(x, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(x * WIDTH, 0);
  ctx.lineTo(x * WIDTH, HEIGHT);
  ctx.stroke();
  ctx.setLineDash([]);
}

function box(b, color, label) {
  if (!b) return;
  const w = b.w * WIDTH, h = b.h * HEIGHT;
  const x = b.x * WIDTH - w / 2, y = b.y * HEIGHT - h / 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.font = '16px sans-serif';
  ctx.fillText(`${label} ${b.conf.toFixed(2)}`, x, Math.max(16, y - 6));
}
