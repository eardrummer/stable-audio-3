// Optimized Canvas Copy-Shift Spectrogram Drawer.
// Scrolls the existing image down 2px and paints the new slice at the top, so
// time falls downward while frequency runs left-to-right.
// Shared by the studio (App.tsx) and the live app (live/LiveApp.tsx).
//
// `x`/`w` confine the whole thing to a vertical column of the canvas, which is
// how the live app shows one lane per variant side by side: each lane scrolls
// its own strip, and the pixels between lanes are never touched.
//
// `shift` is how far the image moves per call. The studio leaves it at the
// historical 2px per frame; the live app drives it from the audio clock
// instead, so a given number of pixels always means the same number of
// seconds and a fixed playback line can be trusted.
export const drawSpectrogram = (
  canvas: HTMLCanvasElement,
  analyser: AnalyserNode,
  colorTheme: 'cyan' | 'green',
  opts?: { x?: number; w?: number; shift?: number }
) => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const x0 = opts?.x ?? 0;
  const width = opts?.w ?? canvas.width;
  const height = canvas.height;
  const shift = Math.max(1, Math.round(opts?.shift ?? 2));
  if (width < 1) return;

  // 1. Scroll this column down
  ctx.drawImage(canvas, x0, 0, width, height, x0, shift, width, height);

  // 2. Capture and draw the new top slice
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  const sliceWidth = width / bufferLength;

  for (let i = 0; i < bufferLength; i++) {
    const value = dataArray[i]; // 0 - 255
    const percent = value / 255.0;

    if (value > 0) {
      let color = '';
      if (colorTheme === 'cyan') {
        // Cyan theme: deep blue/purple to glowing cyan
        const r = 0;
        const g = Math.round(percent * 240);
        const b = Math.round(150 + percent * 105);
        color = `rgba(${r}, ${g}, ${b}, ${percent * 0.45})`; // Semi-transparent glow
      } else {
        // Green theme: deep forest jade to glowing neon green
        const r = 0;
        const g = Math.round(180 + percent * 75);
        const b = Math.round(percent * 102);
        color = `rgba(${r}, ${g}, ${b}, ${percent * 0.45})`; // Semi-transparent glow
      }
      ctx.fillStyle = color;
    } else {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    }
    // The +1 covers rounding gaps between bins, but must not bleed into the
    // neighbouring lane, so the last bin is clipped to the column.
    const bx = x0 + i * sliceWidth;
    ctx.fillRect(bx, 0, Math.min(sliceWidth + 1, x0 + width - bx), shift);
  }
};
