/**
 * AuraTone AI Studio - Client Logic
 * Handles interactive piano roll rendering, audio playback synchronization,
 * parameter adjustments, generation requests, and training analytics.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentGenre = 'classical';
  let currentTrackData = null;
  let isPlaying = false;
  let animationFrameId = null;

  // DOM Elements
  const genreBtns = document.querySelectorAll('.genre-btn');
  const tempSlider = document.getElementById('temp-slider');
  const tempVal = document.getElementById('temp-val');
  const tempDesc = document.getElementById('temp-description');
  const notesSlider = document.getElementById('notes-slider');
  const notesVal = document.getElementById('notes-val');
  const bpmSlider = document.getElementById('bpm-slider');
  const bpmVal = document.getElementById('bpm-val');
  const generateBtn = document.getElementById('generate-btn');

  const playbackSection = document.getElementById('playback-section');
  const trackName = document.getElementById('track-name');
  const trackMeta = document.getElementById('track-meta');
  const playPauseBtn = document.getElementById('play-pause-btn');
  const playIcon = document.getElementById('play-icon');
  const nativeAudio = document.getElementById('native-audio');
  const audioProgressBar = document.getElementById('audio-progress-bar');
  const progressFill = document.getElementById('progress-fill');
  const currentTimeText = document.getElementById('current-time-text');
  const totalTimeText = document.getElementById('total-time-text');
  const downloadMidiBtn = document.getElementById('download-midi-btn');
  const downloadWavBtn = document.getElementById('download-wav-btn');

  // Tabs
  const tabPianoRoll = document.getElementById('tab-pianoroll');
  const tabTokens = document.getElementById('tab-tokens');
  const tabAnalytics = document.getElementById('tab-analytics');
  const viewPianoRoll = document.getElementById('view-pianoroll');
  const viewTokens = document.getElementById('view-tokens');
  const viewAnalytics = document.getElementById('view-analytics');

  const canvas = document.getElementById('pianoRollCanvas');
  const ctx = canvas.getContext('2d');
  const canvasEmpty = document.getElementById('canvas-empty');
  const tokenChipsWrapper = document.getElementById('token-chips-wrapper');

  const lossCanvas = document.getElementById('lossCanvas');
  const lossCtx = lossCanvas.getContext('2d');
  const triggerTrainBtn = document.getElementById('trigger-train-btn');

  // 1. Genre Selection
  genreBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      genreBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentGenre = btn.dataset.genre;
      if (currentGenre === 'jazz') {
        bpmSlider.value = 115;
        bpmVal.textContent = '115 BPM';
      } else {
        bpmSlider.value = 100;
        bpmVal.textContent = '100 BPM';
      }
      loadAnalytics(currentGenre);
    });
  });

  // 2. Temperature feedback
  tempSlider.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    tempVal.textContent = v.toFixed(2);
    if (v < 0.5) {
      tempDesc.textContent = 'Strict Classical Discipline';
    } else if (v <= 1.0) {
      tempDesc.textContent = 'Harmonic Sweet-spot';
    } else if (v <= 1.4) {
      tempDesc.textContent = 'Syncopated Jazz Expression';
    } else {
      tempDesc.textContent = 'Avant-Garde Free Improvisation';
    }
  });

  // 3. Notes & BPM Sliders
  notesSlider.addEventListener('input', (e) => {
    notesVal.textContent = e.target.value;
  });

  bpmSlider.addEventListener('input', (e) => {
    bpmVal.textContent = `${e.target.value} BPM`;
  });

  // 4. Tab Navigation
  function switchTab(activeTab, activeView) {
    [tabPianoRoll, tabTokens, tabAnalytics].forEach(t => t.classList.remove('active'));
    [viewPianoRoll, viewTokens, viewAnalytics].forEach(v => v.style.display = 'none');
    activeTab.classList.add('active');
    activeView.style.display = 'block';

    if (activeTab === tabPianoRoll && currentTrackData) {
      drawPianoRoll();
    } else if (activeTab === tabAnalytics) {
      loadAnalytics(currentGenre);
    }
  }

  tabPianoRoll.addEventListener('click', () => switchTab(tabPianoRoll, viewPianoRoll));
  tabTokens.addEventListener('click', () => switchTab(tabTokens, viewTokens));
  tabAnalytics.addEventListener('click', () => switchTab(tabAnalytics, viewAnalytics));

  // 5. Generate Composition Request
  generateBtn.addEventListener('click', async () => {
    generateBtn.classList.add('loading');
    generateBtn.disabled = true;

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genre: currentGenre,
          temperature: parseFloat(tempSlider.value),
          num_notes: parseInt(notesSlider.value),
          bpm: parseInt(bpmSlider.value)
        })
      });

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Generation failed');
      }

      currentTrackData = data;
      setupPlayback(data);
      renderTokens(data.tokens);
      drawPianoRoll();
      canvasEmpty.style.display = 'none';

      // Switch to Piano Roll view
      switchTab(tabPianoRoll, viewPianoRoll);

    } catch (err) {
      alert(`Error generating music: ${err.message}`);
    } finally {
      generateBtn.classList.remove('loading');
      generateBtn.disabled = false;
    }
  });

  // 6. Audio Playback Setup
  function setupPlayback(data) {
    playbackSection.style.display = 'block';
    trackName.textContent = `${data.genre.toUpperCase()} AI Opus (${data.num_notes} elements)`;
    trackMeta.textContent = `${data.num_notes} Notes & Chords • ${data.total_duration}s • ${data.bpm} BPM • Temp: ${data.temperature}`;

    nativeAudio.src = data.wav_url;
    nativeAudio.load();

    downloadMidiBtn.href = data.midi_url;
    downloadMidiBtn.setAttribute('download', `${data.genre}_ai_opus.mid`);

    downloadWavBtn.href = data.wav_url;
    downloadWavBtn.setAttribute('download', `${data.genre}_ai_opus.wav`);

    progressFill.style.width = '0%';
    currentTimeText.textContent = '0:00';
    totalTimeText.textContent = formatTime(data.total_duration);
    setIsPlaying(false);
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  function setIsPlaying(playing) {
    isPlaying = playing;
    playIcon.textContent = isPlaying ? '❚❚' : '▶';
    if (isPlaying) {
      animatePianoRoll();
    } else {
      cancelAnimationFrame(animationFrameId);
      drawPianoRoll();
    }
  }

  playPauseBtn.addEventListener('click', () => {
    if (!nativeAudio.src) return;
    if (isPlaying) {
      nativeAudio.pause();
      setIsPlaying(false);
    } else {
      nativeAudio.play().then(() => setIsPlaying(true)).catch(console.error);
    }
  });

  nativeAudio.addEventListener('ended', () => {
    setIsPlaying(false);
    progressFill.style.width = '0%';
    currentTimeText.textContent = '0:00';
    drawPianoRoll();
  });

  nativeAudio.addEventListener('timeupdate', () => {
    if (!nativeAudio.duration) return;
    const progress = (nativeAudio.currentTime / nativeAudio.duration) * 100;
    progressFill.style.width = `${progress}%`;
    currentTimeText.textContent = formatTime(nativeAudio.currentTime);
  });

  audioProgressBar.addEventListener('click', (e) => {
    if (!nativeAudio.duration) return;
    const rect = audioProgressBar.getBoundingClientRect();
    const clickRatio = (e.clientX - rect.left) / rect.width;
    nativeAudio.currentTime = clickRatio * nativeAudio.duration;
    drawPianoRoll();
  });

  // 7. Interactive Piano Roll Canvas Rendering
  function drawPianoRoll(currentTime = null) {
    if (!currentTrackData || !currentTrackData.events) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const events = currentTrackData.events.filter(e => e.type !== 'rest');
    if (events.length === 0) return;

    const totalDur = currentTrackData.total_duration || 20;

    // Determine Pitch Range
    const midis = events.map(e => e.midi);
    const minMidi = Math.max(24, Math.min(...midis) - 3);
    const maxMidi = Math.min(100, Math.max(...midis) + 3);
    const midiRange = Math.max(12, maxMidi - minMidi + 1);

    const padLeft = 45;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 25;

    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;
    const laneHeight = plotHeight / midiRange;

    // Draw Background Grid & Piano Keys
    for (let m = minMidi; m <= maxMidi; m++) {
      const y = padTop + (maxMidi - m) * laneHeight;
      const isBlackKey = [1, 3, 6, 8, 10].includes(m % 12);

      // Key lane background
      ctx.fillStyle = isBlackKey ? 'rgba(15, 20, 32, 0.95)' : 'rgba(22, 29, 46, 0.6)';
      ctx.fillRect(padLeft, y, plotWidth, laneHeight);

      // Lane divider line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotWidth, y);
      ctx.stroke();

      // Pitch Label on Left (C, D, E...)
      if (m % 12 === 0) { // Octave C
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const octave = Math.floor(m / 12) - 1;
        ctx.fillText(`C${octave}`, padLeft - 6, y + laneHeight / 2);

        // Highlight C lines
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.25)';
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(padLeft + plotWidth, y);
        ctx.stroke();
      }
    }

    // Measure Vertical Time Bars
    const beatsTotal = Math.ceil(totalDur * (currentTrackData.bpm / 60));
    for (let b = 0; b <= beatsTotal; b++) {
      const t = b * (60 / currentTrackData.bpm);
      const x = padLeft + (t / totalDur) * plotWidth;
      if (x > padLeft + plotWidth) break;

      ctx.strokeStyle = (b % 4 === 0) ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = (b % 4 === 0) ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotHeight);
      ctx.stroke();
    }

    // Active playback time
    const playheadTime = (currentTime !== null) ? currentTime : (nativeAudio.currentTime || 0);

    // Draw Note Blocks
    events.forEach(ev => {
      const x = padLeft + (ev.start / totalDur) * plotWidth;
      const w = Math.max(4, (ev.duration / totalDur) * plotWidth - 2);
      const y = padTop + (maxMidi - ev.midi) * laneHeight;
      const h = Math.max(3, laneHeight - 2);

      const isActive = playheadTime >= ev.start && playheadTime <= (ev.start + ev.duration);

      // Gradient & Color based on pitch and active state
      let grad = ctx.createLinearGradient(x, y, x + w, y + h);
      if (isActive) {
        // Glowing active note
        ctx.shadowColor = '#06b6d4';
        ctx.shadowBlur = 15;
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(1, '#22d3ee');
      } else {
        ctx.shadowBlur = 0;
        if (ev.type === 'chord_note') {
          grad.addColorStop(0, '#06b6d4');
          grad.addColorStop(1, '#0284c7');
        } else if (ev.midi % 12 === 0) { // Tonic root
          grad.addColorStop(0, '#a78bfa');
          grad.addColorStop(1, '#7c3aed');
        } else {
          grad.addColorStop(0, '#f59e0b');
          grad.addColorStop(1, '#d97706');
        }
      }

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 3);
      ctx.fill();

      // Pitch text inside wider notes
      if (w > 26 && h > 9) {
        ctx.fillStyle = isActive ? '#000000' : '#ffffff';
        ctx.font = '9px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ev.pitch, x + w / 2, y + h / 2);
      }
    });

    ctx.shadowBlur = 0;

    // Draw Playhead Line
    const playheadX = padLeft + (playheadTime / totalDur) * plotWidth;
    if (playheadX >= padLeft && playheadX <= padLeft + plotWidth) {
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#f43f5e';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(playheadX, padTop - 4);
      ctx.lineTo(playheadX, padTop + plotHeight + 4);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Playhead triangle cursor at top
      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.moveTo(playheadX - 6, padTop - 4);
      ctx.lineTo(playheadX + 6, padTop - 4);
      ctx.lineTo(playheadX, padTop + 4);
      ctx.closePath();
      ctx.fill();
    }
  }

  function animatePianoRoll() {
    if (!isPlaying) return;
    drawPianoRoll(nativeAudio.currentTime);
    animationFrameId = requestAnimationFrame(animatePianoRoll);
  }

  // 8. Render Tokens Chips
  function renderTokens(tokens) {
    tokenChipsWrapper.innerHTML = '';
    tokens.forEach((tok, idx) => {
      const chip = document.createElement('span');
      chip.className = 'token-chip';
      if (tok.includes('.')) {
        chip.classList.add('chord');
      } else if (tok.startsWith('rest')) {
        chip.classList.add('rest');
      }
      chip.textContent = `#${idx + 1}: ${tok}`;
      tokenChipsWrapper.appendChild(chip);
    });
  }

  // 9. Analytics & Loss Curve Visualization
  async function loadAnalytics(genre) {
    try {
      const [histRes, statsRes] = await Promise.all([
        fetch(`/api/history/${genre}`),
        fetch('/api/stats')
      ]);

      const history = await histRes.json();
      const stats = await statsRes.json();

      if (stats.classical) {
        document.getElementById('stat-classical-tokens').textContent = stats.classical.tokens.toLocaleString();
        document.getElementById('stat-vocab').textContent = `${stats.classical.vocab_size} Classes`;
        document.getElementById('classical-status-text').textContent = stats.classical.model_trained ? 'Ready' : 'Not Trained';
      }
      if (stats.jazz) {
        document.getElementById('stat-jazz-tokens').textContent = stats.jazz.tokens.toLocaleString();
        document.getElementById('jazz-status-text').textContent = stats.jazz.model_trained ? 'Ready' : 'Not Trained';
      }

      if (history.loss && history.loss.length > 0) {
        document.getElementById('stat-loss').textContent = history.loss[history.loss.length - 1];
        drawLossCurve(history.epoch, history.loss);
      }
    } catch (err) {
      console.warn('Analytics load error:', err);
    }
  }

  function drawLossCurve(epochs, losses) {
    const width = lossCanvas.width;
    const height = lossCanvas.height;
    lossCtx.clearRect(0, 0, width, height);

    if (!epochs || epochs.length === 0) return;

    const pad = 40;
    const plotW = width - pad * 2;
    const plotH = height - pad * 2;

    const maxLoss = Math.max(...losses, 1.0);
    const minLoss = 0.0;

    // Draw Grid Lines
    lossCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    lossCtx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
      const y = pad + (plotH / 4) * i;
      lossCtx.beginPath();
      lossCtx.moveTo(pad, y);
      lossCtx.lineTo(pad + plotW, y);
      lossCtx.stroke();

      const labelVal = (maxLoss - (i / 4) * maxLoss).toFixed(2);
      lossCtx.fillStyle = '#64748b';
      lossCtx.font = '10px JetBrains Mono';
      lossCtx.textAlign = 'right';
      lossCtx.fillText(labelVal, pad - 8, y + 3);
    }

    // Draw Smooth Loss Curve
    lossCtx.beginPath();
    const points = epochs.map((ep, i) => {
      const x = pad + ((i) / (epochs.length - 1 || 1)) * plotW;
      const y = pad + (1 - (losses[i] - minLoss) / (maxLoss - minLoss)) * plotH;
      return { x, y };
    });

    // Stroke Gradient
    const grad = lossCtx.createLinearGradient(pad, 0, pad + plotW, 0);
    grad.addColorStop(0, '#8b5cf6');
    grad.addColorStop(1, '#06b6d4');

    lossCtx.strokeStyle = grad;
    lossCtx.lineWidth = 3;
    lossCtx.shadowColor = 'rgba(6, 182, 212, 0.5)';
    lossCtx.shadowBlur = 10;

    points.forEach((pt, idx) => {
      if (idx === 0) lossCtx.moveTo(pt.x, pt.y);
      else {
        const prev = points[idx - 1];
        const cx = (prev.x + pt.x) / 2;
        lossCtx.quadraticCurveTo(prev.x, prev.y, cx, (prev.y + pt.y) / 2);
      }
    });
    const last = points[points.length - 1];
    lossCtx.lineTo(last.x, last.y);
    lossCtx.stroke();
    lossCtx.shadowBlur = 0;

    // Fill underneath curve
    lossCtx.lineTo(last.x, pad + plotH);
    lossCtx.lineTo(points[0].x, pad + plotH);
    lossCtx.closePath();
    const fillGrad = lossCtx.createLinearGradient(0, pad, 0, pad + plotH);
    fillGrad.addColorStop(0, 'rgba(139, 92, 246, 0.25)');
    fillGrad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
    lossCtx.fillStyle = fillGrad;
    lossCtx.fill();

    // Data Point Dots
    points.forEach((pt, i) => {
      lossCtx.fillStyle = '#ffffff';
      lossCtx.beginPath();
      lossCtx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      lossCtx.fill();
    });
  }

  // 10. Trigger Model Retrain
  triggerTrainBtn.addEventListener('click', async () => {
    triggerTrainBtn.disabled = true;
    triggerTrainBtn.textContent = 'Training...';

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genre: currentGenre, epochs: 15, samples: 450 })
      });
      const data = await res.json();
      if (data.success) {
        alert(`Training complete for ${currentGenre}! Final Loss: ${data.final_loss}`);
        loadAnalytics(currentGenre);
      } else {
        alert(`Training error: ${data.error}`);
      }
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      triggerTrainBtn.disabled = false;
      triggerTrainBtn.textContent = 'Retrain Model';
    }
  });

  // Initial Load
  loadAnalytics('classical');
});
