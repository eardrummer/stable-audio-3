// Optimized Canvas Copy-Shift Spectrogram Drawer.
// Scrolls the existing image down 2px and paints the new slice at the top, so
// time falls downward while frequency runs left-to-right.
// Shared by the studio (App.tsx) and the live app (live/LiveApp.tsx).
export const drawSpectrogram = (
  canvas: HTMLCanvasElement,
  analyser: AnalyserNode,
  colorTheme: 'cyan' | 'green'
) => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;

  // 1. Scroll canvas down by 2px
  ctx.drawImage(canvas, 0, 0, width, height, 0, 2, width, height);

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
      ctx.fillRect(i * sliceWidth, 0, sliceWidth + 1, 2);
    } else {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(i * sliceWidth, 0, sliceWidth + 1, 2);
    }
  }
};
