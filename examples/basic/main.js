/**
 * Basic example: renders a sine-wave matrix with interactive controls.
 */
import init, { LeibnizFast } from '../../pkg/leibniz_fast.js';

const canvas = document.getElementById('canvas');
const colormapSelect = document.getElementById('colormap');
const sizeSelect = document.getElementById('size');
const tooltip = document.getElementById('tooltip');

/**
 * Generate a sine-wave test matrix.
 * Creates interesting visual patterns for testing zoom/pan/colormap.
 *
 * @param {number} rows
 * @param {number} cols
 * @returns {Float32Array}
 */
function generateSineWave(rows, cols) {
  const data = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c / cols;
      const y = r / rows;
      // Combine multiple frequencies for visual interest
      data[r * cols + c] =
        Math.sin(x * 20) * Math.cos(y * 20) +
        Math.sin((x + y) * 10) * 0.5 +
        Math.sin(Math.sqrt(x * x + y * y) * 30) * 0.3;
    }
  }
  return data;
}

async function main() {
  // Initialize WASM module
  await init();

  // Create the viewer
  const viewer = await LeibnizFast.create(canvas, colormapSelect.value);

  // Generate and set initial data
  let size = parseInt(sizeSelect.value);
  let data = generateSineWave(size, size);
  viewer.setData(data, size, size);

  // Set up hover tooltip
  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });

  // Mouse interaction: pan (drag) and hover (tooltip)
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    viewer.onMouseDown(e.clientX - rect.left, e.clientY - rect.top);
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    viewer.onMouseMove(e.clientX - rect.left, e.clientY - rect.top);
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top = `${e.clientY + 12}px`;
  });

  window.addEventListener('mouseup', () => {
    viewer.onMouseUp();
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  // Scroll to zoom (cursor-anchored)
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    viewer.onWheel(e.clientX - rect.left, e.clientY - rect.top, -e.deltaY);
  }, { passive: false });

  // Colormap change
  colormapSelect.addEventListener('change', () => {
    viewer.setColormap(colormapSelect.value);
  });

  // Size change
  sizeSelect.addEventListener('change', () => {
    size = parseInt(sizeSelect.value);
    data = generateSineWave(size, size);
    viewer.setData(data, size, size);
  });
}

main().catch(console.error);
