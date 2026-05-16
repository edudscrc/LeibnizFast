import { LeibnizFast } from '../../dist/index.js';

const canvas = document.getElementById('line-canvas');
const tooltip = document.getElementById('tooltip');
const errorBanner = document.getElementById('error-banner');
const modeSelect = document.getElementById('mode');
const gridInput = document.getElementById('grid');
const reloadBtn = document.getElementById('reload');

const SAMPLES = 1200;
const STREAM_NEW_SAMPLES = 6;

/** @type {LeibnizFast | null} */
let viewer = null;
let animationId = 0;
let phase = 0;
let streamTotal = SAMPLES;
let pointerInPlot = false;

function formatUnit(unit) {
  return unit ? ` ${unit}` : '';
}

function isPointerInPlot(event) {
  const rect = canvas.getBoundingClientRect();
  return (
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom
  );
}

function hideTooltip() {
  pointerInPlot = false;
  tooltip.style.display = 'none';
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.style.display = 'block';
}

function makeSeries(sampleCount, phaseOffset) {
  const signal = new Float32Array(sampleCount);
  const envelope = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    const t = (i / (sampleCount - 1)) * Math.PI * 8;
    signal[i] = Math.sin(t + phaseOffset) + 0.25 * Math.sin(t * 3.1);
    envelope[i] = 0.55 * Math.cos(t * 0.5 + phaseOffset * 0.35);
  }
  return [
    { id: 'signal', name: 'Signal', color: [80, 190, 255, 1], data: signal },
    { id: 'envelope', name: 'Envelope', color: [255, 170, 64, 1], data: envelope },
  ];
}

function makeStreamingUpdate(startSample, count) {
  const signal = new Float32Array(count);
  const envelope = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const sample = startSample + i;
    const t = sample * 0.025;
    signal[i] = Math.sin(t) + 0.25 * Math.sin(t * 3.1);
    envelope[i] = 0.55 * Math.cos(t * 0.5);
  }
  return [
    { id: 'signal', data: signal },
    { id: 'envelope', data: envelope },
  ];
}

async function createViewer() {
  if (viewer) viewer.destroy();
  const support = await LeibnizFast.checkSupport();
  if (!support.supported) {
    showError(support.reason || 'WebGPU is required for LeibnizFast.');
    return;
  }

  viewer = await LeibnizFast.create(canvas, {
    chart: {
      type: 'line',
      title: 'Oscilloscope Trace',
      grid: gridInput.checked,
      xAxis: { kind: 'linear', label: 'Time', unit: 's', min: 0, max: 2 },
      yAxis: { label: 'Amplitude', unit: 'V', rangeMode: 'stickyAuto' },
      backgroundColor: '#151821',
      labelColor: '#d8dee9',
      tickColor: '#6b7280',
    },
  });

  viewer.onHover((info) => {
    if (info.kind !== 'line') return;
    if (!pointerInPlot) {
      tooltip.style.display = 'none';
      return;
    }

    tooltip.style.display = 'block';
    const rows = info.points
      .map((point) => {
        const [r, g, b, a] = point.color;
        return `<div class="tooltip-row"><span class="tooltip-swatch" style="background: rgba(${r}, ${g}, ${b}, ${a})"></span><span>${point.seriesName}</span><strong>${point.value.toFixed(4)}${formatUnit(info.yUnit)}</strong></div>`;
      })
      .join('');
    tooltip.innerHTML = `<div class="tooltip-x">X: ${info.x.toFixed(3)}${formatUnit(info.xUnit)}</div>${rows}`;
  });
}

document.addEventListener(
  'mousemove',
  (event) => {
    pointerInPlot = isPointerInPlot(event);
    if (!pointerInPlot) {
      tooltip.style.display = 'none';
      return;
    }
    tooltip.style.left = `${event.clientX + 12}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
  },
  { capture: true },
);
canvas.addEventListener('mouseleave', hideTooltip);
document.addEventListener('mouseleave', hideTooltip);
window.addEventListener('blur', hideTooltip);

function stopLoop() {
  if (animationId !== 0) {
    cancelAnimationFrame(animationId);
    animationId = 0;
  }
}

function runStatic() {
  viewer.setLineData(makeSeries(SAMPLES, 0), {
    xAxis: { kind: 'linear', label: 'Time', unit: 's', min: 0, max: 2 },
    resetYRange: true,
  });
}

function runAnimated() {
  const loop = () => {
    phase += 0.035;
    viewer.setLineData(makeSeries(SAMPLES, phase), {
      xAxis: { kind: 'linear', label: 'Time', unit: 's', min: 0, max: 2 },
    });
    animationId = requestAnimationFrame(loop);
  };
  loop();
}

function runStreaming() {
  streamTotal = SAMPLES;
  viewer.setChart({
    type: 'line',
    title: 'Streaming Oscilloscope',
    grid: gridInput.checked,
    xAxis: { kind: 'streaming', label: 'Time', unit: 's', unitsPerSample: 0.002 },
    yAxis: { label: 'Amplitude', unit: 'V', rangeMode: 'stickyAuto' },
    backgroundColor: '#151821',
    labelColor: '#d8dee9',
    tickColor: '#6b7280',
  });
  viewer.setLineData(makeSeries(SAMPLES, 0), { xOffset: streamTotal, resetYRange: true });

  const loop = () => {
    const updates = makeStreamingUpdate(streamTotal, STREAM_NEW_SAMPLES);
    streamTotal += STREAM_NEW_SAMPLES;
    viewer.setLineDataScrolled(updates, {
      newSamples: STREAM_NEW_SAMPLES,
      xOffset: streamTotal,
    });
    animationId = requestAnimationFrame(loop);
  };
  loop();
}

async function load() {
  stopLoop();
  await createViewer();
  if (!viewer) return;

  const mode = modeSelect.value;
  if (mode === 'animated') runAnimated();
  else if (mode === 'streaming') runStreaming();
  else runStatic();
}

reloadBtn.addEventListener('click', () => load().catch((error) => showError(String(error))));
modeSelect.addEventListener('change', () => load().catch((error) => showError(String(error))));
gridInput.addEventListener('change', () => load().catch((error) => showError(String(error))));

load().catch((error) => showError(String(error)));
