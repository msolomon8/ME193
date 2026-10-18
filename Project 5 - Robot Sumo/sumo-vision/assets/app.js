// Shows the board's marked-up camera picture and asks you to approve what it finds.
//
// The board sends "view" about 5 times a second: the picture plus its state. When it has a
// question (opponent? / we're out?), the question box shows up; Approve / Reject (or Enter /
// Delete on the keyboard) send an "answer" back. "We're out?" wins if both are open.

const socket = io(window.location.origin);

const statusEl = document.querySelector('#status');
const viewEl = document.querySelector('#view');
const readout = document.querySelector('#readout');
const questionBox = document.querySelector('#question');
const questionText = document.querySelector('#question-text');
const stateEl = document.querySelector('#state');

let question = null;   // "out", "opponent" or null

socket.on('connect', () => { statusEl.textContent = 'Connected'; });
socket.on('disconnect', () => { statusEl.textContent = 'Disconnected'; });

socket.on('view', (v) => {
  viewEl.src = 'data:image/jpeg;base64,' + v.jpg;

  if (v.out_state === 'asking') question = 'out';   // the laptop stopped: back sensor + camera both saw tape
  else if (v.opp_state === 'asking') question = 'opponent';
  else question = null;

  if (question === 'out') questionText.textContent = "Robot stopped: back sensor and camera both saw the tape. We're OUT? (Approve = sad song, Reject = keep going)";
  if (question === 'opponent') questionText.textContent = 'Is the green outline the OPPONENT? (charge)';
  questionBox.className = question ? question : 'hidden';

  if (v.out_state === 'sent') stateEl.textContent = 'OUT - sad song sent to the laptop';
  else if (v.opp_state === 'approved') stateEl.textContent = 'CHARGING (approved)';
  else if (v.learning) stateEl.textContent = 'Learning the floor color... keep the camera on empty floor';
  else stateEl.textContent = 'Waiting';
  stateEl.className = v.out_state === 'sent' ? 'out' : (v.opp_state === 'approved' ? 'charging' : '');

  readout.textContent =
    `opponent: ${v.opp_seen ? 'seen' : 'no'}  size=${v.opp_size.toFixed(2)}   ` +
    `tape ahead=${v.tape_ahead.toFixed(2)}`;
});

function answer(action) {
  socket.emit('answer', { action });
}

function approve() {
  if (question) answer('approve_' + question);
}

function reject() {
  if (question) answer('reject_' + question);
}

document.querySelector('#approve-btn').addEventListener('click', approve);
document.querySelector('#reject-btn').addEventListener('click', reject);
document.querySelector('#stop-btn').addEventListener('click', () => answer('stop_charge'));
document.querySelector('#out-btn').addEventListener('click', () => answer('out_now'));
document.querySelector('#floor-btn').addEventListener('click', () => answer('relearn_floor'));

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); approve(); }
  if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); reject(); }
});
