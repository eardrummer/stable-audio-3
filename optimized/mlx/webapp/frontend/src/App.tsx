import React, { useState, useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js';
import {
  Play,
  Pause,
  Mic,
  Square,
  Upload,
  Sparkles,
  Sliders,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Music,
  Download,
  Plus,
  History,
  X,
  Repeat,
  Volume2,
  SkipBack,
  Grid,
  Activity,
  RefreshCw,
  Radio
} from 'lucide-react';

// API Base URL
const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

interface AudioItem {
  id: string;
  timestamp: number;
  prompt: string;
  negative_prompt?: string;
  init_audio_path?: string;
  init_noise_level: number;
  inpaint_start?: number;
  inpaint_end?: number;
  seconds: number;
  steps: number;
  seed?: number;
  dit_name: string;
  decoder_name: string;
  cfg: number;
  apg: number;
  filename: string;
  wav_url: string;
}

interface SeedVariant {
  seed: number;
  wav_url: string;
  filename: string;
  isReady: boolean;
}

interface VariantItem {
  noise_level: number;
  delta: number;
  seeds: SeedVariant[]; // Array of 5 seed variations per noise level
}

// Extract the concrete seed used from a filename (e.g. out_12345.wav -> 12345)
const intFromFilename = (filename: string): number => {
  try {
    const parts = filename.split(".wav")[0].split("_");
    return parseInt(parts[parts.length - 1], 10);
  } catch (e) {
    return 0;
  }
};

// Optimized Canvas Copy-Shift Spectrogram Drawer
const drawSpectrogram = (
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

// Fast Client-Side Float32 PCM to 16-bit Stereo WAV Encoder
function encodePcmToWav(left: Float32Array, right: Float32Array, sampleRate: number): Blob {
  const numChannels = 2;
  const numSamples = left.length;
  const buffer = new ArrayBuffer(44 + numSamples * numChannels * 2);
  const view = new DataView(buffer);

  function writeString(offset: number, string: string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // RIFF header
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * numChannels * 2, true);
  writeString(8, 'WAVE');

  // fmt chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, 1, true);  // AudioFormat (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true); // ByteRate
  view.setUint16(32, numChannels * 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample

  // data chunk
  writeString(36, 'data');
  view.setUint32(40, numSamples * numChannels * 2, true);

  // Write 16-bit PCM samples
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const sL = Math.max(-1, Math.min(1, left[i] || 0));
    const sR = Math.max(-1, Math.min(1, right[i] || 0));

    const valL = sL < 0 ? sL * 0x8000 : sL * 0x7FFF;
    const valR = sR < 0 ? sR * 0x8000 : sR * 0x7FFF;

    view.setInt16(offset, valL, true);
    offset += 2;
    view.setInt16(offset, valR, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

export default function App() {
  // Unique connection ID for websocket routing
  const [connectionId] = useState(() => Math.random().toString(36).substring(2, 15));
  
  // Prompt States
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [useNegPrompt, setUseNegPrompt] = useState(false);

  // Audio Source Mode Tabs ('single' | 'blend' | 'live')
  const [sourceMode, setSourceMode] = useState<'single' | 'blend' | 'live'>('single');
  const [blendRatio, setBlendRatio] = useState<number>(0.5); // A vs B blend (0.0 to 1.0)
  const [blendMethod, setBlendMethod] = useState<'waveform' | 'latent_linear' | 'latent_slerp' | 'asymmetric_noise' | 'vector_field'>('waveform');

  // Live Source States
  const [audioInputDevices, setAudioInputDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [selectedChannel, setSelectedChannel] = useState<'stereo' | 'ch1' | 'ch2'>('stereo');
  const [isLiveFeedActive, setIsLiveFeedActive] = useState<boolean>(false);
  const [liveBufferTargetSecs, setLiveBufferTargetSecs] = useState<number>(10); // 3 | 10 | 20 seconds
  const [liveBufferDuration, setLiveBufferDuration] = useState<number>(0);
  const [liveMonitorVolume, setLiveMonitorVolume] = useState<number>(0); // 0.0 default (muted)
  const [livePeakLevel, setLivePeakLevel] = useState<number>(0);

  // Audio Source (Input 1) States
  const [initAudio, setInitAudio] = useState<{
    filename: string;
    local_path: string;
    duration: number;
    wav_url: string;
  } | null>(null);
  const [activeSourceUrl, setActiveSourceUrl] = useState<string | null>(null);
  const [isSourcePlaying, setIsSourcePlaying] = useState(false);
  const [sourceProgress, setSourceProgress] = useState(0);
  const [sourceVolume, setSourceVolume] = useState(1.0); // 0.0 - 1.0
  const [inpaintRange, setInpaintRange] = useState<[number, number] | null>(null);

  // Audio Source (Input 2) States
  const [initAudio2, setInitAudio2] = useState<{
    filename: string;
    local_path: string;
    duration: number;
    wav_url: string;
  } | null>(null);
  const [activeSourceUrl2, setActiveSourceUrl2] = useState<string | null>(null);
  const [isSourcePlaying2, setIsSourcePlaying2] = useState(false);
  const [sourceProgress2, setSourceProgress2] = useState(0);
  const [sourceVolume2, setSourceVolume2] = useState(1.0); // 0.0 - 1.0
  
  // Generated Audio (Output) States
  const [generatedAudio, setGeneratedAudio] = useState<{
    filename: string;
    duration: number;
    wav_url: string;
  } | null>(null);
  const [activeOutputUrl, setActiveOutputUrl] = useState<string | null>(null);
  const [isOutputPlaying, setIsOutputPlaying] = useState(false);
  const [isOutputLooping, setIsOutputLooping] = useState(true); // Output loop active by default!
  const [outputTrimRange, setOutputTrimRange] = useState<[number, number] | null>(null); // Trimming cursors
  const [outputProgress, setOutputProgress] = useState(0);
  const [outputVolume, setOutputVolume] = useState(1.0); // 0.0 - 1.0
  
  // 2D Fine Control Noise Variants (21 noise levels, each with 5 seed variations)
  const [variants, setVariants] = useState<VariantItem[]>([]);
  const [activeVariantIndex, setActiveVariantIndex] = useState<number>(10); // Index 10 is delta = 0.0 (Center)
  const [activeSeedIndex, setActiveSeedIndex] = useState<number>(0); // Index 0 is primary generation
  
  // Global Tape Deck Transport States
  const [isGlobalPlaying, setIsGlobalPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(true); // Loop active by default!
  const [masterVolume, setMasterVolume] = useState(0.8); // Master volume defaults to 80%
  
  // Recording States
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordingTarget, setRecordingTarget] = useState<1 | 2>(1); // Which source is recording
  const [audioUploadLoading, setAudioUploadLoading] = useState(false);
  const [audioUploadLoading2, setAudioUploadLoading2] = useState(false);
  
  // Generation Settings
  const [ditName, setDitName] = useState<'sm-music' | 'sm-sfx' | 'medium'>('sm-music');
  const [isModelMissing, setIsModelMissing] = useState(false);
  const [isPreloading, setIsPreloading] = useState(false);
  const [preloadProgressText, setPreloadProgressText] = useState('');
  const [noiseLevel, setNoiseLevel] = useState(0.3); // 0.05 - 1.0 (for Generate A)
  const [cfg, setCfg] = useState(4.0);
  const [steps, setSteps] = useState(8);
  const [seed, setSeed] = useState<number | ''>('');
  const [extendLength, setExtendLength] = useState(10); // seconds (for Generate B)
  const [freeModels, setFreeModels] = useState(false);
  
  // App UI State
  const [activeTab, setActiveTab] = useState<'generate_a' | 'generate_b'>('generate_a');
  const [saveHistory, setSaveHistory] = useState<boolean>(() => {
    const saved = localStorage.getItem('saveHistory');
    return saved !== null ? JSON.parse(saved) : true;
  });

  useEffect(() => {
    localStorage.setItem('saveHistory', JSON.stringify(saveHistory));
  }, [saveHistory]);

  // 2D Matrix Generation Monitor States
  const [showMonitor, setShowMonitor] = useState<boolean>(false);
  const [generatingCell, setGeneratingCell] = useState<{vIndex: number; sIndex: number} | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [progressTotal, setProgressTotal] = useState(8);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressText, setProgressText] = useState('Initializing pipeline...');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  
  // History
  const [history, setHistory] = useState<AudioItem[]>([]);
  const [modelStatus, setModelStatus] = useState<any>(null);

  // Refs for audio/recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef2 = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Wavesurfer Refs
  const sourceWavesurferRef = useRef<WaveSurfer | null>(null);
  const sourceContainerRef = useRef<HTMLDivElement>(null);
  const sourceRegionsRef = useRef<any>(null);

  const sourceWavesurferRef2 = useRef<WaveSurfer | null>(null);
  const sourceContainerRef2 = useRef<HTMLDivElement>(null);

  const outputWavesurferRef = useRef<WaveSurfer | null>(null);
  const outputContainerRef = useRef<HTMLDivElement>(null);
  const outputRegionsRef = useRef<any>(null);
  const isOutputLoopingRef = useRef<boolean>(true);
  const outputTrimRangeRef = useRef<[number, number] | null>(null);

  // Playhead synchronization & Smooth Crossfade refs
  const pendingPlayheadRef = useRef<{ time: number; play: boolean; fadeIn?: boolean } | null>(null);
  const outgoingAudioRef = useRef<HTMLAudioElement | null>(null);

  // Spectrogram Canvas Refs
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);

  // Web Audio Context & Analyser Refs (Source)
  const sourceAudioCtxRef = useRef<AudioContext | null>(null);
  const sourceAnalyserRef = useRef<AnalyserNode | null>(null);
  const connectedSourceMediaRef = useRef<HTMLMediaElement | null>(null);
  const sourceAnimationRef = useRef<number | null>(null);

  // Web Audio Context & Analyser Refs (Source 2)
  const sourceAudioCtxRef2 = useRef<AudioContext | null>(null);
  const sourceAnalyserRef2 = useRef<AnalyserNode | null>(null);
  const connectedSourceMediaRef2 = useRef<HTMLMediaElement | null>(null);
  const sourceAnimationRef2 = useRef<number | null>(null);

  // Live Source Web Audio Refs & 10s Ring Buffer
  const liveAudioCtxRef = useRef<AudioContext | null>(null);
  const liveStreamRef = useRef<MediaStream | null>(null);
  const liveSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const liveGainNodeRef = useRef<GainNode | null>(null);
  const liveAnalyserRef = useRef<AnalyserNode | null>(null);
  const liveProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const liveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveAnimationRef = useRef<number | null>(null);
  const liveRingBufferRef = useRef<{
    left: Float32Array;
    right: Float32Array;
    writeIndex: number;
    totalSamplesRecorded: number;
    sampleRate: number;
  } | null>(null);

  // Web Audio Context & Analyser Refs (Output)
  const outputAudioCtxRef = useRef<AudioContext | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const connectedOutputMediaRef = useRef<HTMLMediaElement | null>(null);
  const outputAnimationRef = useRef<number | null>(null);

  // Keep a ref of looping state for wavesurfer event listeners
  const isLoopingRef = useRef(isLooping);
  useEffect(() => {
    isLoopingRef.current = isLooping;
  }, [isLooping]);

  // 1. Setup WebSocket & Load History on Mount
  useEffect(() => {
    fetchHistory();
    fetchModelStatus();
    
    const ws = new WebSocket(`${WS_URL}/ws/${connectionId}`);
    ws.onopen = () => console.log('[WS] Connected with ID:', connectionId);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
          const step = data.step;
          const total = data.total;
          setProgressStep(step);
          setProgressTotal(total);
          setProgressPercent(Math.round((step / total) * 100));
          setProgressText(`Denoising: Step ${step} of ${total} (${Math.round(data.elapsed_ms)}ms)`);
        } else if (data.type === 'complete') {
          setProgressPercent(100);
          setProgressText('Generation complete! Decoding audio...');
          
          const actualSeed = intFromFilename(data.filename);

          // Populate the Center (CTR) Variant 0
          setVariants(prev => {
            return prev.map((v, idx) => {
              if (idx === 10) {
                const updatedSeeds = [...v.seeds];
                updatedSeeds[0] = {
                  seed: actualSeed,
                  wav_url: `${API_URL}${data.wav_url}`,
                  filename: data.filename,
                  isReady: true
                };
                return { ...v, seeds: updatedSeeds };
              }
              return v;
            });
          });

          // Load into Generated Output player
          setGeneratedAudio({
            filename: data.filename,
            duration: data.seconds,
            wav_url: `${API_URL}${data.wav_url}`
          });
          setActiveOutputUrl(`${API_URL}${data.wav_url}`);
          setActiveSeedIndex(0); // Reset to seed index 0
          setActiveVariantIndex(0); // Reset to base 0.0
          fetchHistory();
        } else if (data.type === 'variant_start') {
          // Background variant generation started!
          const { delta, seed_index } = data;
          setVariants(prev => {
            const vIdx = prev.findIndex(v => Math.abs(v.delta - delta) < 1e-4);
            if (vIdx !== -1) {
              setGeneratingCell({ vIndex: vIdx, sIndex: seed_index });
            }
            return prev;
          });
        } else if (data.type === 'variant_ready') {
          // Background variant generation completed!
          const { delta, filename, wav_url, seed_index } = data;
          
          setVariants(prev => {
            const vIdx = prev.findIndex(v => Math.abs(v.delta - delta) < 1e-4);
            
            // Clear generating cell if it matches this completed slot
            if (vIdx !== -1) {
              setGeneratingCell(curr => {
                if (curr && curr.vIndex === vIdx && curr.sIndex === seed_index) {
                  return null;
                }
                return curr;
              });
            }
            
            return prev.map((v, idx) => {
              if (idx === vIdx) {
                const updatedSeeds = [...v.seeds];
                const actualSeed = intFromFilename(filename);
                updatedSeeds[seed_index] = {
                  seed: actualSeed,
                  wav_url: `${API_URL}${wav_url}`,
                  filename: filename,
                  isReady: true
                };
                return { ...v, seeds: updatedSeeds };
              }
              return v;
            });
          });
        } else if (data.type === 'queue_complete') {
          setIsGenerating(false);
          setProgressPercent(100);
          setProgressText('All background variations generated successfully!');
        } else if (data.type === 'error') {
          setErrorMsg(data.error);
          setIsGenerating(false);
        }
      } catch (e) {}
    };
    
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 30000);

    return () => {
      ws.close();
      clearInterval(pingInterval);
      if (sourceWavesurferRef.current) sourceWavesurferRef.current.destroy();
      if (outputWavesurferRef.current) outputWavesurferRef.current.destroy();
    };
  }, [connectionId]);

  // 2. Load Source Wavesurfer when active source URL changes
  useEffect(() => {
    if (!activeSourceUrl || !sourceContainerRef.current) return;

    if (sourceWavesurferRef.current) {
      sourceWavesurferRef.current.destroy();
      sourceWavesurferRef.current = null;
    }

    const ws = WaveSurfer.create({
      container: sourceContainerRef.current,
      waveColor: 'rgba(255, 255, 255, 0.12)',
      progressColor: '#00f0ff',
      cursorColor: '#ffffff',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 3,
      height: 80,
      normalize: true,
      fillParent: true
    });

    sourceWavesurferRef.current = ws;
    ws.setVolume(sourceVolume * masterVolume);

    // Register Regions Plugin
    const regions = ws.registerPlugin(RegionsPlugin.create());
    sourceRegionsRef.current = regions;

    regions.enableDragSelection({
      color: 'rgba(0, 240, 255, 0.15)',
    });

    regions.on('region-created', (region) => {
      regions.getRegions().forEach((r: any) => {
        if (r.id !== region.id) r.destroy();
      });
      setInpaintRange([Number(region.start.toFixed(2)), Number(region.end.toFixed(2))]);
    });

    regions.on('region-updated', (region) => {
      setInpaintRange([Number(region.start.toFixed(2)), Number(region.end.toFixed(2))]);
    });

    ws.on('ready', () => {
      setIsSourcePlaying(false);
      setSourceProgress(0);
    });

    ws.on('audioprocess', () => {
      const current = ws.getCurrentTime();
      const dur = ws.getDuration();
      setSourceProgress(dur > 0 ? (current / dur) * 100 : 0);
    });

    ws.on('finish', () => {
      if (isLoopingRef.current) {
        ws.play();
      } else {
        setIsSourcePlaying(false);
        setIsGlobalPlaying(false);
        setSourceProgress(0);
      }
    });

    ws.load(activeSourceUrl);

    return () => {
      if (ws) {
        ws.destroy();
        sourceWavesurferRef.current = null;
      }
    };
  }, [activeSourceUrl, isGenerating]);

  // 2b. Load Source 2 Wavesurfer when active source URL 2 changes
  useEffect(() => {
    if (!activeSourceUrl2 || !sourceContainerRef2.current) return;

    if (sourceWavesurferRef2.current) {
      sourceWavesurferRef2.current.destroy();
      sourceWavesurferRef2.current = null;
    }

    const ws = WaveSurfer.create({
      container: sourceContainerRef2.current,
      waveColor: 'rgba(255, 255, 255, 0.12)',
      progressColor: '#00f0ff',
      cursorColor: '#ffffff',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 3,
      height: 80,
      normalize: true,
      fillParent: true
    });

    sourceWavesurferRef2.current = ws;
    ws.setVolume(sourceVolume2 * masterVolume);

    ws.on('ready', () => {
      setIsSourcePlaying2(false);
      setSourceProgress2(0);
    });

    ws.on('audioprocess', () => {
      const current = ws.getCurrentTime();
      const dur = ws.getDuration();
      setSourceProgress2(dur > 0 ? (current / dur) * 100 : 0);
    });

    ws.on('finish', () => {
      if (isLoopingRef.current) {
        ws.play();
      } else {
        setIsSourcePlaying2(false);
        setIsGlobalPlaying(false);
        setSourceProgress2(0);
      }
    });

    ws.load(activeSourceUrl2);

    return () => {
      if (ws) {
        ws.destroy();
        sourceWavesurferRef2.current = null;
      }
    };
  }, [activeSourceUrl2, isGenerating]);

  // 3. Load Output Wavesurfer when active output URL changes
  useEffect(() => {
    if (!activeOutputUrl || !outputContainerRef.current) return;

    if (outputWavesurferRef.current) {
      outputWavesurferRef.current.destroy();
      outputWavesurferRef.current = null;
    }

    const ws = WaveSurfer.create({
      container: outputContainerRef.current,
      waveColor: 'rgba(255, 255, 255, 0.12)',
      progressColor: '#00ff66',
      cursorColor: '#ffffff',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 3,
      height: 80,
      normalize: true,
      fillParent: true
    });

    outputWavesurferRef.current = ws;
    ws.setVolume(outputVolume * masterVolume);

    // Register Regions Plugin on Output for Trimming Handles & Boundaries
    const regions = ws.registerPlugin(RegionsPlugin.create());
    outputRegionsRef.current = regions;

    regions.on('region-updated', (region: any) => {
      const range: [number, number] = [Number(region.start.toFixed(2)), Number(region.end.toFixed(2))];
      setOutputTrimRange(range);
      outputTrimRangeRef.current = range;
    });

    ws.on('ready', () => {
      setIsOutputPlaying(false);
      setOutputProgress(0);

      const duration = ws.getDuration();
      const prevTrim = outputTrimRangeRef.current;
      let startTrim = 0;
      let endTrim = duration;
      if (prevTrim) {
        startTrim = Math.min(prevTrim[0], duration);
        endTrim = Math.min(prevTrim[1], duration);
        if (endTrim <= startTrim || endTrim === 0) endTrim = duration;
      }
      regions.clearRegions();
      regions.addRegion({
        start: startTrim,
        end: endTrim,
        color: 'rgba(0, 240, 255, 0.08)',
        drag: false,
        resize: true
      });
      setOutputTrimRange([Number(startTrim.toFixed(2)), Number(endTrim.toFixed(2))]);
      outputTrimRangeRef.current = [Number(startTrim.toFixed(2)), Number(endTrim.toFixed(2))];

      // Synced Playhead & Smooth 350ms Equal-Power Crossfade Restoration!
      if (pendingPlayheadRef.current) {
        const { time, play, fadeIn } = pendingPlayheadRef.current;
        const seekTime = time < duration ? time : 0;
        ws.setTime(seekTime);
        
        if (play) {
          const targetVol = outputVolume * masterVolume;
          if (fadeIn) {
            // Smooth 350ms equal-power sine fade-in to blend seamlessly with outgoing audio
            ws.setVolume(0);
            ws.play();
            setIsOutputPlaying(true);

            const startFade = Date.now();
            const fadeDur = 350;
            const stepFadeIn = () => {
              const elapsed = Date.now() - startFade;
              const prog = Math.min(1, elapsed / fadeDur);
              // Equal power sine curve: sin(prog * pi / 2)
              const gain = Math.sin(prog * 0.5 * Math.PI);
              ws.setVolume(targetVol * gain);
              if (prog < 1) {
                requestAnimationFrame(stepFadeIn);
              } else {
                ws.setVolume(targetVol);
              }
            };
            requestAnimationFrame(stepFadeIn);
          } else {
            ws.setVolume(targetVol);
            ws.play();
            setIsOutputPlaying(true);
          }

          // If global synced play was active, resume source too
          if (isGlobalPlaying && sourceWavesurferRef.current && !isSourcePlaying) {
            sourceWavesurferRef.current.play();
            setIsSourcePlaying(true);
          }
        }
        pendingPlayheadRef.current = null;
      }
    });

    ws.on('audioprocess', () => {
      const current = ws.getCurrentTime();
      const dur = ws.getDuration();
      setOutputProgress(dur > 0 ? (current / dur) * 100 : 0);

      // Trimming boundary checks!
      const trim = outputTrimRangeRef.current;
      if (trim && current >= trim[1]) {
        if (isOutputLoopingRef.current || isLoopingRef.current) {
          ws.setTime(trim[0]);
          ws.play();
        } else {
          ws.pause();
          ws.setTime(trim[0]);
          setIsOutputPlaying(false);
          setOutputProgress(dur > 0 ? (trim[0] / dur) * 100 : 0);
        }
      }
    });

    ws.on('finish', () => {
      const trim = outputTrimRangeRef.current;
      const startSeek = trim ? trim[0] : 0;
      if (isOutputLoopingRef.current || isLoopingRef.current) {
        ws.setTime(startSeek);
        ws.play();
      } else {
        setIsOutputPlaying(false);
        setIsGlobalPlaying(false);
        ws.setTime(startSeek);
        setOutputProgress(0);
      }
    });

    ws.load(activeOutputUrl);

    return () => {
      if (ws) {
        ws.destroy();
        outputWavesurferRef.current = null;
      }
    };
  }, [activeOutputUrl, isGenerating]);

  // Web Audio Binder (Source Column)
  const setupSourceWebAudio = (mediaElement: HTMLMediaElement) => {
    let audioCtx = sourceAudioCtxRef.current;
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      sourceAudioCtxRef.current = audioCtx;
    }
    
    if (connectedSourceMediaRef.current === mediaElement) return;
    
    try {
      const sourceNode = audioCtx.createMediaElementSource(mediaElement);
      let analyserNode = sourceAnalyserRef.current;
      if (!analyserNode) {
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 512; // 256 frequency bins
        sourceAnalyserRef.current = analyserNode;
      }
      
      sourceNode.connect(analyserNode);
      analyserNode.connect(audioCtx.destination);
      connectedSourceMediaRef.current = mediaElement;
    } catch (e) {
      console.warn("[WebAudio] Failed to bind source media element:", e);
    }
  };

  // Web Audio Binder (Source 2)
  const setupSourceWebAudio2 = (mediaElement: HTMLMediaElement) => {
    let audioCtx = sourceAudioCtxRef2.current;
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      sourceAudioCtxRef2.current = audioCtx;
    }
    
    if (connectedSourceMediaRef2.current === mediaElement) return;
    
    try {
      const sourceNode = audioCtx.createMediaElementSource(mediaElement);
      let analyserNode = sourceAnalyserRef2.current;
      if (!analyserNode) {
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 512; // 256 frequency bins
        sourceAnalyserRef2.current = analyserNode;
      }
      
      sourceNode.connect(analyserNode);
      analyserNode.connect(audioCtx.destination);
      connectedSourceMediaRef2.current = mediaElement;
    } catch (e) {
      console.warn("[WebAudio] Failed to bind source 2 media element:", e);
    }
  };

  // Web Audio Binder (Output Column)
  const setupOutputWebAudio = (mediaElement: HTMLMediaElement) => {
    let audioCtx = outputAudioCtxRef.current;
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      outputAudioCtxRef.current = audioCtx;
    }
    
    if (connectedOutputMediaRef.current === mediaElement) return;
    
    try {
      const sourceNode = audioCtx.createMediaElementSource(mediaElement);
      let analyserNode = outputAnalyserRef.current;
      if (!analyserNode) {
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 512; // 256 frequency bins
        outputAnalyserRef.current = analyserNode;
      }
      
      sourceNode.connect(analyserNode);
      analyserNode.connect(audioCtx.destination);
      connectedOutputMediaRef.current = mediaElement;
    } catch (e) {
      console.warn("[WebAudio] Failed to bind output media element:", e);
    }
  };

  // 4. Source Spectrogram Animation Loop
  useEffect(() => {
    const canvas = sourceCanvasRef.current;
    const ws = sourceWavesurferRef.current;
    
    if (!canvas || !ws || !isSourcePlaying) {
      if (sourceAnimationRef.current) {
        cancelAnimationFrame(sourceAnimationRef.current);
        sourceAnimationRef.current = null;
      }
      return;
    }
    
    const media = ws.getMediaElement();
    if (media) {
      setupSourceWebAudio(media);
    }
    
    const analyser = sourceAnalyserRef.current;
    if (!analyser) return;
    
    if (sourceAudioCtxRef.current && sourceAudioCtxRef.current.state === 'suspended') {
      sourceAudioCtxRef.current.resume();
    }
    
    const render = () => {
      if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
      }
      drawSpectrogram(canvas, analyser, 'cyan');
      sourceAnimationRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (sourceAnimationRef.current) {
        cancelAnimationFrame(sourceAnimationRef.current);
        sourceAnimationRef.current = null;
      }
    };
  }, [isSourcePlaying, activeSourceUrl]);

  // 4b. Source 2 Spectrogram Animation Loop
  useEffect(() => {
    const canvas = sourceCanvasRef.current;
    const ws = sourceWavesurferRef2.current;
    
    if (!canvas || !ws || !isSourcePlaying2) {
      if (sourceAnimationRef2.current) {
        cancelAnimationFrame(sourceAnimationRef2.current);
        sourceAnimationRef2.current = null;
      }
      return;
    }
    
    const media = ws.getMediaElement();
    if (media) {
      setupSourceWebAudio2(media);
    }
    
    const analyser = sourceAnalyserRef2.current;
    if (!analyser) return;
    
    if (sourceAudioCtxRef2.current && sourceAudioCtxRef2.current.state === 'suspended') {
      sourceAudioCtxRef2.current.resume();
    }
    
    const render = () => {
      if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
      }
      drawSpectrogram(canvas, analyser, 'cyan');
      sourceAnimationRef2.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (sourceAnimationRef2.current) {
        cancelAnimationFrame(sourceAnimationRef2.current);
        sourceAnimationRef2.current = null;
      }
    };
  }, [isSourcePlaying2, activeSourceUrl2]);

  // 5. Output Spectrogram Animation Loop
  useEffect(() => {
    const canvas = outputCanvasRef.current;
    const ws = outputWavesurferRef.current;
    
    if (!canvas || !ws || !isOutputPlaying) {
      if (outputAnimationRef.current) {
        cancelAnimationFrame(outputAnimationRef.current);
        outputAnimationRef.current = null;
      }
      return;
    }
    
    const media = ws.getMediaElement();
    if (media) {
      setupOutputWebAudio(media);
    }
    
    const analyser = outputAnalyserRef.current;
    if (!analyser) return;
    
    if (outputAudioCtxRef.current && outputAudioCtxRef.current.state === 'suspended') {
      outputAudioCtxRef.current.resume();
    }
    
    const render = () => {
      if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
      }
      drawSpectrogram(canvas, analyser, 'green');
      outputAnimationRef.current = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (outputAnimationRef.current) {
        cancelAnimationFrame(outputAnimationRef.current);
        outputAnimationRef.current = null;
      }
    };
  }, [isOutputPlaying, activeOutputUrl]);

  // API Calls
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/api/history`);
      const data = await res.json();
      setHistory(data);
    } catch (e) {
      console.error('Failed to fetch history:', e);
    }
  };

  const fetchModelStatus = async (targetDit?: string) => {
    try {
      const res = await fetch(`${API_URL}/api/status`);
      const data = await res.json();
      setModelStatus(data.models);

      const activeDit = targetDit || ditName;
      const ditReady = data.models?.[activeDit]?.downloaded;
      const t5Ready = data.models?.['t5gemma']?.downloaded;

      if (ditReady === false || t5Ready === false) {
        setIsModelMissing(true);
      } else {
        setIsModelMissing(false);
      }
    } catch (e) {
      console.error('Failed to fetch model status:', e);
    }
  };

  const handlePreloadModel = async () => {
    setIsPreloading(true);
    setPreloadProgressText(`Downloading weights for ${ditName.toUpperCase()} into local MLX cache...`);
    try {
      const res = await fetch(`${API_URL}/api/preload_model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dit_name: ditName })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to download model weights');
      }
      const data = await res.json();
      setModelStatus(data.models);
      setIsModelMissing(false);
      setIsPreloading(false);
    } catch (e: any) {
      console.error('Preload failed:', e);
      setPreloadProgressText('Error: ' + e.message);
      setIsPreloading(false);
    }
  };

  useEffect(() => {
    fetchModelStatus(ditName);
  }, [ditName]);

  // Drag & Drop / File Upload Handlers
  const [dragOver, setDragOver] = useState(false);
  const [dragOver2, setDragOver2] = useState(false);
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };
  
  const handleDragLeave = () => {
    setDragOver(false);
  };
  
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await uploadFile(file, 1);
    }
  };

  const handleDragOver2 = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver2(true);
  };
  
  const handleDragLeave2 = () => {
    setDragOver2(false);
  };
  
  const handleDrop2 = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver2(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await uploadFile(file, 2);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>, target: 1 | 2) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await uploadFile(file, target);
    }
  };

  const uploadFile = async (file: File, target: 1 | 2) => {
    if (target === 1) setAudioUploadLoading(true);
    else setAudioUploadLoading2(true);
    setErrorMsg(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to upload audio');
      }
      
      const data = await res.json();
      const payload = {
        filename: data.filename,
        local_path: data.local_path,
        duration: data.duration,
        wav_url: `${API_URL}${data.wav_url}`
      };

      if (target === 1) {
        setInitAudio(payload);
        setActiveSourceUrl(`${API_URL}${data.wav_url}`);
        setInpaintRange(null);
      } else {
        setInitAudio2(payload);
        setActiveSourceUrl2(`${API_URL}${data.wav_url}`);
      }
    } catch (e: any) {
      setErrorMsg(e.message);
    } finally {
      if (target === 1) setAudioUploadLoading(false);
      else setAudioUploadLoading2(false);
    }
  };

  // Mic Recording
  const startRecording = async (target: 1 | 2) => {
    setErrorMsg(null);
    audioChunksRef.current = [];
    setRecordingTime(0);
    setRecordingTarget(target);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const file = new File([audioBlob], `mic_recording_${target}.webm`, { type: 'audio/webm' });
        await uploadFile(file, target);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start(250);
      setIsRecording(true);
      
      recordTimerRef.current = setInterval(() => {
        setRecordingTime(t => t + 1);
      }, 1000);

      // Real-time Oscilloscope using Web Audio API
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      setTimeout(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const canvasCtx = canvas.getContext('2d');
        
        const draw = () => {
          if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
            audioContext.close();
            return;
          }
          
          requestAnimationFrame(draw);
          analyser.getByteTimeDomainData(dataArray);
          
          if (canvasCtx && canvas) {
            canvasCtx.fillStyle = '#000000';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
            
            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = '#ff3b30';
            canvasCtx.shadowBlur = 4;
            canvasCtx.shadowColor = '#ff3b30';
            
            canvasCtx.beginPath();
            const sliceWidth = canvas.width * 1.0 / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
              const v = dataArray[i] / 128.0;
              const y = v * canvas.height / 2;
              
              if (i === 0) {
                canvasCtx.moveTo(x, y);
              } else {
                canvasCtx.lineTo(x, y);
              }
              x += sliceWidth;
            }
            
            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();
          }
        };
        draw();
      }, 100);
      
    } catch (e: any) {
      console.error(e);
      setErrorMsg('Microphone access denied or not available.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordTimerRef.current);
    }
  };

  // Live Audio Input Device Enumeration
  const refreshAudioDevices = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter(d => d.kind === 'audioinput');
      setAudioInputDevices(inputs);
      if (inputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(inputs[0].deviceId);
      }
    } catch (e) {
      console.warn("Failed to enumerate audio devices:", e);
    }
  };

  useEffect(() => {
    refreshAudioDevices();
    const handleDeviceChange = () => refreshAudioDevices();
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
      return () => {
        navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      };
    }
  }, []);

  // Cleanup Live Feed on unmount
  useEffect(() => {
    return () => {
      stopLiveFeed();
    };
  }, []);

  // Live Feed Start / Stop & Oscilloscope
  const startLiveFeed = async () => {
    setErrorMsg(null);
    stopLiveFeed();

    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      liveAudioCtxRef.current = audioCtx;
      const sampleRate = audioCtx.sampleRate || 44100;

      const constraints: MediaStreamConstraints = {
        audio: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      liveStreamRef.current = stream;

      // Refresh labels now that permission is granted
      refreshAudioDevices();

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      liveSourceNodeRef.current = sourceNode;

      // Monitor Gain Node (Muted / 0.0 by default)
      const gainNode = audioCtx.createGain();
      gainNode.gain.value = liveMonitorVolume;
      liveGainNodeRef.current = gainNode;

      sourceNode.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      // Analyser for Live Oscilloscope
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      liveAnalyserRef.current = analyser;
      sourceNode.connect(analyser);

      // Up to 20-Second Circular Ring Buffer Capacity (Supports 3s, 10s, 20s selections)
      const bufferLength = sampleRate * 20;
      const ringBuffer = {
        left: new Float32Array(bufferLength),
        right: new Float32Array(bufferLength),
        writeIndex: 0,
        totalSamplesRecorded: 0,
        sampleRate: sampleRate
      };
      liveRingBufferRef.current = ringBuffer;

      // ScriptProcessor Node to continuously capture PCM audio
      const processor = audioCtx.createScriptProcessor(4096, 2, 2);
      liveProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputL = e.inputBuffer.getChannelData(0);
        const inputR = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : inputL;
        const len = inputL.length;
        const buf = liveRingBufferRef.current;
        if (!buf) return;

        for (let i = 0; i < len; i++) {
          let sL = inputL[i];
          let sR = inputR[i];

          if (selectedChannel === 'ch1') {
            sR = sL;
          } else if (selectedChannel === 'ch2') {
            sL = sR;
          }

          buf.left[buf.writeIndex] = sL;
          buf.right[buf.writeIndex] = sR;
          buf.writeIndex = (buf.writeIndex + 1) % bufferLength;
          buf.totalSamplesRecorded++;
        }

        const recordedSecs = Math.min(20.0, buf.totalSamplesRecorded / sampleRate);
        setLiveBufferDuration(Number(recordedSecs.toFixed(1)));
      };

      sourceNode.connect(processor);
      processor.connect(audioCtx.destination); // Needed to keep processor alive

      setIsLiveFeedActive(true);
      startLiveOscilloscope();
    } catch (err: any) {
      console.error("Live Feed Error:", err);
      setErrorMsg("Failed to start Live Audio Feed: " + err.message);
    }
  };

  const stopLiveFeed = () => {
    if (liveAnimationRef.current) {
      cancelAnimationFrame(liveAnimationRef.current);
      liveAnimationRef.current = null;
    }
    if (liveStreamRef.current) {
      liveStreamRef.current.getTracks().forEach(t => t.stop());
      liveStreamRef.current = null;
    }
    if (liveProcessorRef.current) {
      liveProcessorRef.current.disconnect();
      liveProcessorRef.current = null;
    }
    if (liveGainNodeRef.current) {
      liveGainNodeRef.current.disconnect();
      liveGainNodeRef.current = null;
    }
    if (liveSourceNodeRef.current) {
      liveSourceNodeRef.current.disconnect();
      liveSourceNodeRef.current = null;
    }
    if (liveAudioCtxRef.current) {
      liveAudioCtxRef.current.close().catch(() => {});
      liveAudioCtxRef.current = null;
    }
    setIsLiveFeedActive(false);
    setLivePeakLevel(0);
  };

  const startLiveOscilloscope = () => {
    const draw = () => {
      const canvas = liveCanvasRef.current;
      const analyser = liveAnalyserRef.current;
      if (!canvas || !analyser) {
        liveAnimationRef.current = requestAnimationFrame(draw);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteTimeDomainData(dataArray);

      if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
      }

      ctx.fillStyle = '#060606';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Center grid line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      // Glowing Cyan Waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#00f0ff';
      ctx.shadowBlur = 6;
      ctx.shadowColor = '#00f0ff';
      ctx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      let maxPeak = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const normAmp = Math.abs(dataArray[i] - 128) / 128.0;
        if (normAmp > maxPeak) maxPeak = normAmp;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;

      setLivePeakLevel(maxPeak);
      liveAnimationRef.current = requestAnimationFrame(draw);
    };

    liveAnimationRef.current = requestAnimationFrame(draw);
  };

  const handleLiveMonitorVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = Number(e.target.value);
    setLiveMonitorVolume(vol);
    if (liveGainNodeRef.current && liveAudioCtxRef.current) {
      liveGainNodeRef.current.gain.setValueAtTime(vol, liveAudioCtxRef.current.currentTime);
    }
  };

  const extractLiveBufferAsFile = (): File | null => {
    const buf = liveRingBufferRef.current;
    if (!buf || buf.totalSamplesRecorded === 0) return null;

    const sampleRate = buf.sampleRate;
    const targetSecs = liveBufferTargetSecs;
    const targetLength = sampleRate * targetSecs;
    const finalL = new Float32Array(targetLength);
    const finalR = new Float32Array(targetLength);
    const bufferCapacity = sampleRate * 20;

    if (buf.totalSamplesRecorded < targetLength) {
      // Padded with silence if not yet target length
      const count = Math.min(buf.writeIndex, targetLength);
      for (let i = 0; i < count; i++) {
        finalL[i] = buf.left[i];
        finalR[i] = buf.right[i];
      }
    } else {
      // Unroll the most recent targetLength samples ending at writeIndex
      let readIdx = (buf.writeIndex - targetLength + bufferCapacity) % bufferCapacity;
      for (let i = 0; i < targetLength; i++) {
        finalL[i] = buf.left[readIdx];
        finalR[i] = buf.right[readIdx];
        readIdx = (readIdx + 1) % bufferCapacity;
      }
    }

    const wavBlob = encodePcmToWav(finalL, finalR, sampleRate);
    return new File([wavBlob], `live_snapshot_${targetSecs}s_${Date.now()}.wav`, { type: 'audio/wav' });
  };

  // Real-time Mixer Volume Controls
  const handleSourceVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = Number(e.target.value);
    setSourceVolume(vol);
    if (sourceWavesurferRef.current) {
      sourceWavesurferRef.current.setVolume(vol * masterVolume);
    }
  };

  const handleSourceVolumeChange2 = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = Number(e.target.value);
    setSourceVolume2(vol);
    if (sourceWavesurferRef2.current) {
      sourceWavesurferRef2.current.setVolume(vol * masterVolume);
    }
  };

  const handleOutputVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = Number(e.target.value);
    setOutputVolume(vol);
    if (outputWavesurferRef.current) {
      outputWavesurferRef.current.setVolume(vol * masterVolume);
    }
  };

  const handleMasterVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = Number(e.target.value);
    setMasterVolume(vol);
    if (sourceWavesurferRef.current) {
      sourceWavesurferRef.current.setVolume(sourceVolume * vol);
    }
    if (sourceWavesurferRef2.current) {
      sourceWavesurferRef2.current.setVolume(sourceVolume2 * vol);
    }
    if (outputWavesurferRef.current) {
      outputWavesurferRef.current.setVolume(outputVolume * vol);
    }
  };

  // Synced Play/Pause Control
  const handleGlobalPlayPause = () => {
    const newState = !isGlobalPlaying;
    setIsGlobalPlaying(newState);
    
    const sourceWS = sourceWavesurferRef.current;
    const sourceWS2 = sourceWavesurferRef2.current;
    const outputWS = outputWavesurferRef.current;
    
    if (newState) {
      if (sourceWS && outputWS) {
        if (sourceWS.getCurrentTime() >= sourceWS.getDuration() || 
            outputWS.getCurrentTime() >= outputWS.getDuration()) {
          sourceWS.setTime(0);
          outputWS.setTime(0);
        }
        sourceWS.play();
        outputWS.play();
        setIsSourcePlaying(true);
      }
      if (sourceMode === 'blend' && sourceWS2) {
        if (sourceWS2.getCurrentTime() >= sourceWS2.getDuration()) {
          sourceWS2.setTime(0);
        }
        sourceWS2.play();
        setIsSourcePlaying2(true);
      }
      setIsOutputPlaying(true);
    } else {
      if (sourceWS) {
        sourceWS.pause();
        setIsSourcePlaying(false);
      }
      if (sourceWS2) {
        sourceWS2.pause();
        setIsSourcePlaying2(false);
      }
      if (outputWS) {
        outputWS.pause();
        setIsOutputPlaying(false);
      }
    }
  };

  // Sync Playheads & Return to Zero (Rewind to Start)
  const handleSyncPlayheads = () => {
    const sourceWS = sourceWavesurferRef.current;
    const sourceWS2 = sourceWavesurferRef2.current;
    const outputWS = outputWavesurferRef.current;
    
    if (sourceWS) sourceWS.setTime(0);
    if (sourceWS2) sourceWS2.setTime(0);
    if (outputWS) outputWS.setTime(0);

    // If global sync is rolling, keep them playing together from 0
    if (isGlobalPlaying) {
      if (sourceWS) sourceWS.play();
      if (sourceMode === 'blend' && sourceWS2) sourceWS2.play();
      if (outputWS) outputWS.play();
    }
  };

  // 2D Swapping Logic (Gapless & Synced)
  const loadVariant2D = (vIndex: number, sIndex: number) => {
    setVariants(prev => {
      const variant = prev[vIndex];
      if (!variant) return prev;
      
      let targetSeedIndex = sIndex;
      let seedVar = variant.seeds[targetSeedIndex];
      
      // Fallback: If current seed index is not ready, find the first ready seed variation
      if (!seedVar || !seedVar.isReady) {
        const firstReadyIdx = variant.seeds.findIndex(s => s.isReady);
        if (firstReadyIdx !== -1) {
          targetSeedIndex = firstReadyIdx;
          seedVar = variant.seeds[targetSeedIndex];
        } else {
          return prev; // Truly nothing ready at all
        }
      }
      
      setActiveVariantIndex(vIndex);
      setActiveSeedIndex(targetSeedIndex);
      
      const oldWS = outputWavesurferRef.current;
      if (oldWS) {
        const currentTime = oldWS.getCurrentTime();
        const wasPlaying = isOutputPlaying || isGlobalPlaying;
        
        if (wasPlaying && activeOutputUrl && activeOutputUrl !== seedVar.wav_url) {
          // Smooth 350ms equal-power crossfade with overlap between adjacent files
          try {
            if (outgoingAudioRef.current) {
              outgoingAudioRef.current.pause();
              outgoingAudioRef.current = null;
            }
            const outgoing = new Audio(activeOutputUrl);
            outgoingAudioRef.current = outgoing;
            outgoing.currentTime = currentTime;
            outgoing.volume = outputVolume * masterVolume;
            outgoing.play().catch(() => {});

            const startFade = Date.now();
            const fadeDur = 350;
            const initVol = outputVolume * masterVolume;
            const stepFadeOut = () => {
              const elapsed = Date.now() - startFade;
              const prog = Math.min(1, elapsed / fadeDur);
              if (outgoingAudioRef.current === outgoing) {
                // Equal power cosine curve: cos(prog * pi / 2)
                const gain = Math.cos(prog * 0.5 * Math.PI);
                outgoing.volume = Math.max(0, Math.min(1, initVol * gain));
                if (prog < 1) {
                  requestAnimationFrame(stepFadeOut);
                } else {
                  outgoing.pause();
                  outgoingAudioRef.current = null;
                }
              }
            };
            requestAnimationFrame(stepFadeOut);
          } catch (e) {
            console.warn("Crossfade error:", e);
          }

          pendingPlayheadRef.current = { time: currentTime, play: true, fadeIn: true };
          oldWS.pause();
          setActiveOutputUrl(seedVar.wav_url);
        } else {
          pendingPlayheadRef.current = { time: currentTime, play: wasPlaying, fadeIn: false };
          oldWS.pause();
          setActiveOutputUrl(seedVar.wav_url);
        }
      } else {
        setActiveOutputUrl(seedVar.wav_url);
      }
      
      return prev;
    });
  };

  // Tape Deck Up Arrow -> Cycles to higher seed variations (e.g. 2 -> 3)
  const handleNextVariant = () => {
    const currentLevelSeeds = variants[activeVariantIndex]?.seeds || [];
    for (let i = activeSeedIndex + 1; i < currentLevelSeeds.length; i++) {
      if (currentLevelSeeds[i].isReady) {
        loadVariant2D(activeVariantIndex, i);
        break;
      }
    }
  };

  // Tape Deck Down Arrow -> Cycles to lower seed variations (e.g. 2 -> 1)
  const handlePrevVariant = () => {
    const currentLevelSeeds = variants[activeVariantIndex]?.seeds || [];
    for (let i = activeSeedIndex - 1; i >= 0; i--) {
      if (currentLevelSeeds[i].isReady) {
        loadVariant2D(activeVariantIndex, i);
        break;
      }
    }
  };

  // New Wheel Up Button next to dial -> moves up the wheel (selecting less noise, lower index)
  const handleWheelUp = () => {
    for (let i = activeVariantIndex - 1; i >= 0; i--) {
      if (variants[i]?.seeds.some(s => s.isReady)) {
        loadVariant2D(i, activeSeedIndex);
        break;
      }
    }
  };

  // New Wheel Down Button next to dial -> moves down the wheel (selecting more noise, higher index)
  const handleWheelDown = () => {
    for (let i = activeVariantIndex + 1; i < variants.length; i++) {
      if (variants[i]?.seeds.some(s => s.isReady)) {
        loadVariant2D(i, activeSeedIndex);
        break;
      }
    }
  };

  // Stop background generation immediately
  const handleStopGeneration = async () => {
    try {
      const res = await fetch(`${API_URL}/api/abort`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ connection_id: connectionId })
      });
      if (res.ok) {
        setIsGenerating(false);
        setProgressPercent(0);
        setProgressText('Generation stopped by user.');
      }
    } catch (e) {
      console.error('[Abort] Failed to abort:', e);
    }
  };

  // Trigger Generation
  const handleGenerate = async () => {
    if (sourceMode === 'single') {
      if (!prompt.trim() && !initAudio) {
        setErrorMsg('Please enter a prompt or load an initial audio file.');
        return;
      }
    } else if (sourceMode === 'blend') {
      if (!initAudio || !initAudio2) {
        setErrorMsg('Dual Audio Blend mode requires both Source A and Source B to be uploaded.');
        return;
      }
    } else if (sourceMode === 'live') {
      if (!liveRingBufferRef.current || liveRingBufferRef.current.totalSamplesRecorded === 0) {
        setErrorMsg('Please click "Start Live Feed" to record live microphone audio into the 10s buffer first.');
        return;
      }
    }
    
    setIsGenerating(true);
    setErrorMsg(null);
    setProgressPercent(0);
    setProgressStep(0);
    setProgressTotal(steps);
    setProgressText('Warming up MLX models in unified VRAM...');

    // Initialize the 11 vertical noise levels (Base is 0.0, plus 10 noisier variations)
    const deltas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20];
    const initialVariants = deltas.map(d => {
      const targetNoise = Number((noiseLevel + d).toFixed(2));
      return {
        noise_level: targetNoise,
        delta: Number(d.toFixed(2)),
        seeds: Array.from({ length: 5 }, (_, sIdx) => ({
          seed: 0,
          wav_url: '',
          filename: '',
          isReady: false
        }))
      };
    });
    setVariants(initialVariants);
    setActiveVariantIndex(0); // Base is index 0
    setActiveSeedIndex(0); // Primary is seed index 0

    const payload: any = {
      prompt: prompt,
      negative_prompt: useNegPrompt ? negativePrompt : undefined,
      steps: steps,
      cfg: cfg,
      apg: 1.0,
      seed: seed === '' ? undefined : Number(seed),
      dit_name: ditName,
      decoder_name: decoder_name => undefined, // Unused parameter mapping
      free_models: freeModels,
      connection_id: connectionId,
      save_history: saveHistory  // Support memory toggle!
    };

    if (sourceMode === 'live') {
      try {
        setProgressText('Extracting and uploading 10s live audio buffer snapshot...');
        const snapshotFile = extractLiveBufferAsFile();
        if (!snapshotFile) {
          throw new Error('Failed to capture audio from live 10s buffer.');
        }

        const formData = new FormData();
        formData.append('file', snapshotFile);

        const uploadRes = await fetch(`${API_URL}/api/upload`, {
          method: 'POST',
          body: formData
        });

        if (!uploadRes.ok) {
          const err = await uploadRes.json();
          throw new Error(err.detail || 'Failed to upload live audio buffer');
        }

        const uploadData = await uploadRes.json();
        const uploadedUrl = `${API_URL}${uploadData.wav_url}`;

        setInitAudio({
          filename: uploadData.filename,
          local_path: uploadData.local_path,
          duration: 10.0,
          wav_url: uploadedUrl
        });
        setActiveSourceUrl(uploadedUrl);

        payload.init_audio_path = uploadData.wav_url;
        payload.init_noise_level = noiseLevel;
        payload.seconds = 10.0;
      } catch (err: any) {
        setIsGenerating(false);
        setErrorMsg(err.message || 'Error processing live audio buffer');
        return;
      }
    } else if (sourceMode === 'blend') {
      payload.init_audio_path = initAudio!.wav_url;
      payload.init_audio_path_2 = initAudio2!.wav_url;
      payload.blend_ratio = blendRatio;
      payload.blend_method = blendMethod;
      payload.init_noise_level = noiseLevel;
      payload.seconds = Math.max(initAudio!.duration, initAudio2!.duration);
    } else {
      if (activeTab === 'generate_a') {
        if (initAudio) {
          payload.init_audio_path = initAudio.wav_url;
          payload.init_noise_level = noiseLevel;
          payload.seconds = initAudio.duration;
          
          if (inpaintRange) {
            payload.inpaint_start = inpaintRange[0];
            payload.inpaint_end = inpaintRange[1];
          }
        } else {
          payload.seconds = 30.0;
        }
      } else {
        if (!initAudio) {
          setErrorMsg('Extend option requires an initial audio file.');
          setIsGenerating(false);
          return;
        }
        payload.init_audio_path = initAudio.wav_url;
        payload.init_noise_level = 1.0;
        payload.seconds = initAudio.duration + extendLength;
        payload.inpaint_start = initAudio.duration;
        payload.inpaint_end = initAudio.duration + extendLength;
      }
    }

    try {
      const res = await fetch(`${API_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Generation failed.');
      }
      
      const data = await res.json();
      
      // Load primary output
      setGeneratedAudio({
        filename: data.filename,
        duration: data.seconds,
        wav_url: `${API_URL}${data.wav_url}`
      });
      setActiveOutputUrl(`${API_URL}${data.wav_url}`);
      
      // Update Center node (index 10, seed 0) in the variations matrix to be ready and store its seed!
      setVariants(prev => {
        return prev.map((v, idx) => {
          if (idx === 10) { // Center
            const updatedSeeds = [...v.seeds];
            updatedSeeds[0] = {
              seed: data.variants_matrix?.[0]?.seeds?.[0]?.seed || (seed === '' ? 0 : Number(seed)),
              wav_url: `${API_URL}${data.wav_url}`,
              filename: data.filename,
              isReady: true
            };
            return { ...v, seeds: updatedSeeds };
          }
          return v;
        });
      });

      fetchHistory();
      
      // If no source audio is loaded, there is no background variations queue.
      // We can turn off isGenerating immediately!
      if (!initAudio) {
        setIsGenerating(false);
      }
      
    } catch (e: any) {
      setErrorMsg(e.message);
      setIsGenerating(false); // Reset on failure
    }
  };

  // Trigger on-demand finer variations (Seeds 1 to 4 for active level + neighbors)
  const handleGenerateFinerVariants = async () => {
    if (!generatedAudio || !generatedAudio.filename) {
      setErrorMsg('No active generation session found.');
      return;
    }
    
    // Extract session ID from e.g. "session_uuid/primary.wav"
    const filenameParts = generatedAudio.filename.split('/');
    let sessionId = '';
    for (const part of filenameParts) {
      if (part.startsWith('session_')) {
        sessionId = part.replace('session_', '');
        break;
      }
    }
    
    if (!sessionId) {
      setErrorMsg('Could not resolve session folder.');
      return;
    }

    // Retrieve the actual base seed from the Center node (index 10, seed 0)
    const baseSeed = variants[10]?.seeds[0]?.seed || 0;
    
    setIsGenerating(true);
    setErrorMsg(null);
    setProgressPercent(0);
    setProgressText('Queueing finer variations...');

    const payload = {
      session_id: sessionId,
      connection_id: connectionId,
      current_variant_index: activeVariantIndex,
      center_noise: noiseLevel,
      actual_seed: baseSeed,
      prompt: prompt,
      negative_prompt: useNegPrompt ? negativePrompt : undefined,
      seconds: generatedAudio.duration,
      steps: steps,
      dit_name: ditName,
      decoder_name: 'auto',
      cfg: cfg,
      apg: 1.0,
      free_models: freeModels,
      save_history: saveHistory
    };

    try {
      const res = await fetch(`${API_URL}/api/generate_finer_variants`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Finer variations generation failed.');
      }
      
      const data = await res.json();
      setProgressText(`Queued ${data.task_count} finer variations...`);
      
    } catch (e: any) {
      setErrorMsg(e.message);
      setIsGenerating(false);
    }
  };

  // Playback Control Handlers (Breaking global sync if clicked individually)
  const handlePlayPauseSource = () => {
    if (sourceWavesurferRef.current) {
      sourceWavesurferRef.current.playPause();
      setIsSourcePlaying(!isSourcePlaying);
      setIsGlobalPlaying(false);
    }
  };

  const handlePlayPauseSource2 = () => {
    if (sourceWavesurferRef2.current) {
      sourceWavesurferRef2.current.playPause();
      setIsSourcePlaying2(!isSourcePlaying2);
      setIsGlobalPlaying(false);
    }
  };

  const handlePlayPauseOutput = () => {
    const ws = outputWavesurferRef.current;
    if (!ws) return;
    if (ws.isPlaying()) {
      ws.pause();
      setIsOutputPlaying(false);
    } else {
      const trim = outputTrimRangeRef.current;
      const current = ws.getCurrentTime();
      if (trim && (current < trim[0] || current >= trim[1])) {
        ws.setTime(trim[0]);
      }
      ws.play();
      setIsOutputPlaying(true);
      setIsGlobalPlaying(false);
    }
  };

  const handleClearSourceRegion = () => {
    if (sourceRegionsRef.current) {
      sourceRegionsRef.current.clearRegions();
      setInpaintRange(null);
    }
  };

  // Promote Output -> Source
  const handlePromoteToSource = () => {
    if (!generatedAudio) return;
    
    setInitAudio({
      filename: generatedAudio.filename,
      local_path: '',
      duration: generatedAudio.duration,
      wav_url: generatedAudio.wav_url
    });
    
    setActiveSourceUrl(generatedAudio.wav_url);
    setGeneratedAudio(null);
    setActiveOutputUrl(null);
    setInpaintRange(null);
    setIsGlobalPlaying(false);
    setVariants([]); // Clear variants
  };

  const handleHistoryItemClick = (item: AudioItem & { variants_matrix?: any[] }) => {
    setPrompt(item.prompt);
    if (item.negative_prompt) {
      setNegativePrompt(item.negative_prompt);
      setUseNegPrompt(true);
    } else {
      setUseNegPrompt(false);
    }
    setSteps(item.steps);
    setCfg(item.cfg);
    setDitName(item.dit_name as any);
    
    // Legacy support vs Session subpath support
    const isSessionUrl = item.wav_url.includes("/session_");
    
    setInitAudio({
      filename: item.filename,
      local_path: '',
      duration: item.seconds,
      wav_url: `${API_URL}${item.wav_url}`
    });
    setActiveSourceUrl(`${API_URL}${item.wav_url}`);
    setInpaintRange(null);
    
    setGeneratedAudio({
      filename: isSessionUrl ? "primary.wav" : item.filename,
      duration: item.seconds,
      wav_url: `${API_URL}${item.wav_url}`
    });
    setActiveOutputUrl(`${API_URL}${item.wav_url}`);
    setIsGlobalPlaying(false);
    
    // Recover 2D variants matrix if present!
    if (item.variants_matrix && item.variants_matrix.length > 0) {
      const deltas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20];
      
      const restoredVariants = deltas.map(d => {
        const targetNoise = Number((item.init_noise_level + d).toFixed(2));
        
        // Find saved slot for this delta
        const savedSlot = item.variants_matrix?.find((s: any) => Math.abs(s.delta - d) < 1e-4);
        
        const seeds = Array.from({ length: 5 }, (_, sIdx) => {
          const savedSeed = savedSlot?.seeds?.find((s: any) => s.seed_index === sIdx);
          if (savedSeed && savedSeed.isReady) {
            const sessionFolder = `session_${item.id}`;
            return {
              seed: savedSeed.seed || 0,
              wav_url: `${API_URL}/api/audio/${sessionFolder}/${savedSeed.filename}`,
              filename: savedSeed.filename,
              isReady: true
            };
          }
          return {
            seed: 0,
            wav_url: '',
            filename: '',
            isReady: false
          };
        });

        return {
          noise_level: targetNoise,
          delta: Number(d.toFixed(2)),
          seeds: seeds
        };
      });
      
      setVariants(restoredVariants);
      setActiveVariantIndex(0); // Base is index 0
      setActiveSeedIndex(0); // Primary
    } else {
      setVariants([]); // Clear if no variants saved
    }
    
    setShowHistory(false);
  };

  const handleDeleteHistory = async (e: React.MouseEvent, itemId: string) => {
    e.stopPropagation();
    try {
      await fetch(`${API_URL}/api/history/${itemId}`, { method: 'DELETE' });
      fetchHistory();
      if (activeSourceUrl?.includes(itemId)) {
        setActiveSourceUrl(null);
        setInitAudio(null);
        setInpaintRange(null);
      }
      if (activeOutputUrl?.includes(itemId)) {
        setActiveOutputUrl(null);
        setGeneratedAudio(null);
      }
      setIsGlobalPlaying(false);
      setVariants([]);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="app-container">
      {/* Full-Screen Mirrored Spectrogram Background */}
      <div className="spectrogram-background-container">
        <canvas ref={sourceCanvasRef} className="spectrogram-canvas source-spectrogram" />
        <canvas ref={outputCanvasRef} className="spectrogram-canvas output-spectrogram-mirrored" />
      </div>

      {/* MODEL PRELOAD / DOWNLOAD OVERLAY MODAL */}
      {isModelMissing && (
        <div className="model-download-modal-backdrop">
          <div className="model-download-modal">
            <div className="model-download-icon-wrap">
              <Download size={32} className="icon-glow-cyan" />
            </div>
            <h2 className="model-download-title">MODEL SETUP REQUIRED</h2>
            <p className="model-download-desc">
              The weights for <strong>{ditName.toUpperCase()}</strong> and text encoder (~1.4 GB) are not cached locally yet. 
              This is a one-time automatic download from HuggingFace to your Apple Silicon MLX cache.
            </p>
            
            {isPreloading ? (
              <div className="model-download-loading">
                <div className="progress-loader"></div>
                <span>{preloadProgressText || 'Downloading weights into local cache... Please wait.'}</span>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', width: '100%', alignItems: 'center' }}>
                <button className="btn-primary btn-preload-model" onClick={handlePreloadModel}>
                  <Download size={16} />
                  <span>Download & Load Model Now</span>
                </button>
                <button 
                  className="btn-skip-preload"
                  onClick={() => setIsModelMissing(false)}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', fontSize: '0.75rem', cursor: 'pointer' }}
                >
                  Dismiss (Download on first generate)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Header with Model selector and History Trigger */}
      <header className="app-header">
        <div className="logo">
          <div className="logo-dot"></div>
          SA3 MLX LOCAL <span className="logo-author">by Manaswi</span>
        </div>
        
        <div className="header-controls">
          {/* Model Selection Dropdown */}
          <div className="header-model-select">
            <span className="header-model-label">MODEL</span>
            <select 
              className="model-select-dropdown"
              value={ditName}
              onChange={(e) => setDitName(e.target.value as any)}
            >
              <option value="sm-music">Music (50M)</option>
              <option value="sm-sfx">SFX (50M)</option>
              <option value="medium">Medium (1.4B)</option>
            </select>
          </div>

          <div className="model-status-indicator">
            <div className="status-dot"></div>
            Model cached: {ditName === 'sm-music' ? 'sa3-small (50M)' : ditName === 'sm-sfx' ? 'sa3-sfx (50M)' : 'sa3-medium (1.4B)'}
          </div>



          {/* Memory Toggle Button */}
          <button 
            className={`header-memory-btn ${saveHistory ? 'active' : 'inactive'}`}
            onClick={() => setSaveHistory(!saveHistory)}
            title={saveHistory ? 'Memory Active: Generations are saved to history.' : 'Memory Off: Creations are temporary and auto-deleted on next run.'}
          >
            <div className="memory-indicator-dot"></div>
            {saveHistory ? 'MEMORY ON' : 'MEMORY OFF'}
          </button>

          {/* Toggle History Button */}
          <button 
            className={`header-history-btn ${showHistory ? 'active' : ''}`}
            onClick={() => setShowHistory(true)}
            title="Open Creation History"
          >
            <History size={15} />
            History
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="app-main">
        
        <div className="workspace-wrapper">

          {/* GLOBAL MIXER TRANSPORT BAR (Vintage Tape deck style) */}
          {activeSourceUrl && activeOutputUrl && activeTab === 'generate_a' && (
            <div className="global-transport-deck">
              
              {/* Loop Hardware Button */}
              <button 
                className={`deck-btn btn-loop ${isLooping ? 'active' : ''}`}
                onClick={() => setIsLooping(!isLooping)}
                title={isLooping ? 'Looping Active (Click to disable)' : 'Looping Disabled (Click to enable)'}
              >
                <Repeat size={16} />
              </button>
              
              {/* Return to Start / Sync Playheads Button (RTZ) */}
              <button 
                className="deck-btn btn-sync-playheads"
                onClick={handleSyncPlayheads}
                title="Sync Playheads & Rewind to Start (Return to Zero)"
              >
                <SkipBack size={16} fill="#fff" />
              </button>
              
              {/* Play/Pause Synced Button */}
              <button 
                className={`deck-btn btn-play ${isGlobalPlaying ? 'playing' : ''}`}
                onClick={handleGlobalPlayPause}
                title="Play/Pause both tracks simultaneously in perfect sync"
              >
                {isGlobalPlaying ? <Square size={16} fill="#ff3b30" /> : <Play size={16} fill="#00f0ff" />}
              </button>

              {/* Symmetrical Control Pods */}
              <div className="deck-nav-group">
                {/* Noise Navigation Pod (Vertical, Up/Down) */}
                <div className="deck-nav-subgroup">
                  <button 
                    className="deck-btn btn-nav"
                    onClick={handleWheelUp}
                    disabled={variants.length === 0}
                    title="Scroll Noise Level Up (Y-Axis)"
                  >
                    <ChevronUp size={16} />
                  </button>

                  <div className="deck-lcd-display min-lcd">
                    <span className="lcd-label">NOISE</span>
                    <span className="lcd-value">
                      {variants[activeVariantIndex]?.delta === 0 
                        ? 'CTR' 
                        : (variants[activeVariantIndex]?.delta > 0 ? '+' : '') + variants[activeVariantIndex]?.delta?.toFixed(2)}
                    </span>
                  </div>

                  <button 
                    className="deck-btn btn-nav"
                    onClick={handleWheelDown}
                    disabled={variants.length === 0}
                    title="Scroll Noise Level Down (Y-Axis)"
                  >
                    <ChevronDown size={16} />
                  </button>
                </div>

                {/* Seed Navigation Pod (Horizontal, Left/Right) */}
                <div className="deck-nav-subgroup">
                  <button 
                    className="deck-btn btn-nav"
                    onClick={handlePrevVariant}
                    disabled={variants.length === 0}
                    title="Previous seed variation (X-Axis)"
                  >
                    <ChevronLeft size={16} />
                  </button>
                  
                  {/* Calculate active seed position among ready seeds */}
                  {(() => {
                    const activeRow = variants[activeVariantIndex];
                    let activePosition = 1;
                    let totalReady = 1;
                    if (activeRow) {
                      const readySeeds = activeRow.seeds.filter(s => s.isReady);
                      totalReady = readySeeds.length || 1;
                      const activeSeed = activeRow.seeds[activeSeedIndex];
                      const readyIndex = readySeeds.findIndex(s => s.seed === activeSeed?.seed);
                      if (readyIndex !== -1) {
                        activePosition = readyIndex + 1;
                      }
                    }
                    return (
                      <div className="deck-lcd-display min-lcd">
                        <span className="lcd-label">SEED</span>
                        <span className="lcd-value">{activePosition}/{totalReady}</span>
                      </div>
                    );
                  })()}
                  
                  <button 
                    className="deck-btn btn-nav"
                    onClick={handleNextVariant}
                    disabled={variants.length === 0}
                    title="Next seed variation (X-Axis)"
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>

              {/* Finer Variations Trigger Button */}
              <button
                className={`deck-btn btn-finer-vars ${isGenerating ? 'disabled' : ''}`}
                onClick={handleGenerateFinerVariants}
                disabled={isGenerating || variants.length === 0}
                title="Generate all 5 seed variations for this noise level and its neighbors"
              >
                <Sparkles size={11} style={{ color: 'var(--accent-cyan)' }} />
                <span>FINER VARS</span>
              </button>

              {/* Master Volume Slider Recessed Group */}
              <div className="deck-volume-slider-group">
                <Volume2 size={14} style={{ color: 'var(--text-secondary)' }} />
                <input 
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={masterVolume}
                  onChange={handleMasterVolumeChange}
                  className="slider-input deck-volume-slider"
                  title="Master Mix Volume"
                />
                <span className="deck-volume-text">{Math.round(masterVolume * 100)}%</span>
              </div>

              {/* Minimal Stop/Cancel Generation Button next to Volume */}
              <button
                className={`deck-btn btn-stop-gen ${isGenerating ? 'active' : ''}`}
                onClick={handleStopGeneration}
                disabled={!isGenerating}
                title={isGenerating ? "Stop background variations generation immediately" : "No active generation to stop"}
              >
                <Square size={10} fill={isGenerating ? "var(--accent-red)" : "rgba(255, 255, 255, 0.15)"} style={{ color: isGenerating ? 'var(--accent-red)' : 'rgba(255, 255, 255, 0.15)' }} />
                <span>STOP GEN</span>
              </button>

            </div>
          )}
          
          {/* HORIZONTAL GENERATION PANEL (HARDWARE CONSOLE SPREADING UNDER TRANSPORT) */}
          <div className="horizontal-generation-panel">
            
            {/* Top Row: Large Text Prompt with "prompt guide" placeholder & big text font */}
            <div className="hgen-top-row">
              <div className="hgen-prompt-wrapper">
                <input
                  type="text"
                  className="hgen-prompt-input"
                  placeholder="prompt guide"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
            </div>

            {/* Bottom Row: Controls (Model, Mode, Recessed Sliders, Generate Button) */}
            <div className="hgen-bottom-row">
              <div className="hgen-controls-left">
                {/* Model Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span className="hgen-param-label">MODEL:</span>
                  <select 
                    className="model-select-dropdown"
                    value={ditName}
                    onChange={(e) => setDitName(e.target.value as any)}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.72rem' }}
                  >
                    <option value="sm-music">Music (50M)</option>
                    <option value="sm-sfx">SFX (50M)</option>
                    <option value="medium">Medium (1.4B)</option>
                  </select>
                </div>

                {/* Mode Pill Tabs */}
                <div className="hgen-mode-pills">
                  <button 
                    className={`hgen-mode-pill ${activeTab === 'generate_a' ? 'active' : ''}`}
                    onClick={() => {
                      setActiveTab('generate_a');
                      handleClearSourceRegion();
                    }}
                  >
                    Audio-to-Audio
                  </button>
                  <button 
                    className={`hgen-mode-pill ${activeTab === 'generate_b' ? 'active' : ''}`}
                    onClick={() => {
                      setActiveTab('generate_b');
                      handleClearSourceRegion();
                    }}
                  >
                    Extend Audio
                  </button>
                </div>

                {/* Noise / Variation Slider */}
                {activeTab === 'generate_a' ? (
                  <div className="hgen-param-group">
                    <span className="hgen-param-label">NOISE</span>
                    <input 
                      type="range" 
                      min="0.05" 
                      max="1.0" 
                      step="0.05"
                      value={noiseLevel} 
                      onChange={(e) => setNoiseLevel(Number(e.target.value))}
                      className="slider-input hgen-slider"
                      title={`Noise / Variation Strength: ${Math.round(noiseLevel * 100)}%`}
                    />
                    <span className="hgen-param-badge">{Math.round(noiseLevel * 100)}%</span>
                  </div>
                ) : (
                  <div className="hgen-param-group">
                    <span className="hgen-param-label">LENGTH</span>
                    <input 
                      type="range" 
                      min="5" 
                      max="30" 
                      step="1"
                      value={extendLength} 
                      onChange={(e) => setExtendLength(Number(e.target.value))}
                      className="slider-input hgen-slider"
                      title={`Extension Duration: ${extendLength}s`}
                    />
                    <span className="hgen-param-badge">{extendLength}s</span>
                  </div>
                )}

                {/* Denoising Steps Slider */}
                <div className="hgen-param-group">
                  <span className="hgen-param-label">STEPS</span>
                  <input 
                    type="range" 
                    min="1" 
                    max="32" 
                    step="1"
                    value={steps} 
                    onChange={(e) => setSteps(Number(e.target.value))}
                    className="slider-input hgen-slider"
                    title={`Denoising Steps: ${steps}`}
                  />
                  <span className="hgen-param-badge">{steps}</span>
                </div>

                {/* CFG Slider */}
                <div className="hgen-param-group">
                  <span className="hgen-param-label">CFG</span>
                  <input 
                    type="range" 
                    min="1.0" 
                    max="12.0" 
                    step="0.5"
                    value={cfg} 
                    onChange={(e) => setCfg(Number(e.target.value))}
                    className="slider-input hgen-slider"
                    title={`CFG Guidance: ${cfg}`}
                  />
                  <span className="hgen-param-badge">{cfg}</span>
                </div>

                {/* Advanced Options Toggle */}
                <button 
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: '0.68rem', display: 'flex', alignItems: 'center', gap: '0.2rem', cursor: 'pointer' }}
                  title="Toggle Advanced Seed & VRAM options"
                >
                  {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  <span>ADV</span>
                </button>
              </div>

              {/* Action: Generate Button or Progress Indicator */}
              {isGenerating ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', background: 'rgba(0, 240, 255, 0.05)', border: '1px solid rgba(0, 240, 255, 0.2)', borderRadius: '10px', padding: '0.4rem 1rem' }}>
                  <div className="progress-loader" style={{ width: '18px', height: '18px' }}></div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', fontFamily: 'monospace' }}>
                    {progressText || 'SYNTHESIZING...'} ({progressPercent}%)
                  </span>
                </div>
              ) : (
                <button 
                  className="hgen-btn-generate"
                  onClick={handleGenerate}
                  disabled={isGenerating || audioUploadLoading}
                >
                  <Sparkles size={14} />
                  <span>{activeTab === 'generate_a' ? (inpaintRange ? 'INPAINT REGION' : 'GENERATE AUDIO') : 'EXTEND AUDIO'}</span>
                </button>
              )}
            </div>

            {/* Expandable Advanced Options Drawer */}
            {showAdvanced && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', borderTop: '1px solid rgba(255, 255, 255, 0.04)', paddingTop: '0.6rem', fontSize: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="hgen-param-label">SEED:</span>
                  <input 
                    type="number"
                    placeholder="Random"
                    className="numeric-input"
                    style={{ width: '100px', fontSize: '0.72rem', padding: '0.25rem 0.5rem' }}
                    value={seed}
                    onChange={(e) => setSeed(e.target.value === '' ? '' : Number(e.target.value))}
                  />
                </div>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={freeModels}
                    onChange={(e) => setFreeModels(e.target.checked)}
                    style={{ accentColor: 'var(--accent-cyan)' }}
                  />
                  Free VRAM immediately (8GB Macs)
                </label>
              </div>
            )}

            {/* Error Message */}
            {errorMsg && (
              <div style={{ color: 'var(--accent-red)', fontSize: '0.75rem', padding: '0.4rem 0.8rem', background: 'rgba(255, 59, 48, 0.05)', border: '1px solid rgba(255, 59, 48, 0.15)', borderRadius: '6px' }}>
                {errorMsg}
              </div>
            )}

          </div>
          
          {/* 2-COLUMN WORKSTATION: SOURCE DECK (LEFT) | OUTPUT DECK (RIGHT) */}
          <div className="top-split-container">
            
            {/* LEFT COLUMN: AUDIO SOURCE (INPUT) */}
            <div className="split-column left">
              
              {/* Single Source vs Dual Source Blend vs Live Source Toggle */}
              <div className="parameters-box-header-tabs" style={{ marginBottom: '1rem' }}>
                <button 
                  className={`model-tab-btn ${sourceMode === 'single' ? 'active' : ''}`}
                  onClick={() => setSourceMode('single')}
                >
                  Single Source
                </button>
                <button 
                  className={`model-tab-btn ${sourceMode === 'blend' ? 'active' : ''}`}
                  onClick={() => setSourceMode('blend')}
                >
                  Dual Source Blend
                </button>
                <button 
                  className={`model-tab-btn ${sourceMode === 'live' ? 'active' : ''}`}
                  onClick={() => setSourceMode('live')}
                >
                  Live Source (Mic/Input)
                </button>
              </div>

              {sourceMode === 'single' ? (
                <>
                  {/* Source Volume Slider */}
                  <div className="volume-control-group">
                    <div className="volume-header">
                      <span>Source Volume</span>
                      <span className="volume-value">{Math.round(sourceVolume * 100)}%</span>
                    </div>
                    <input 
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={sourceVolume}
                      onChange={handleSourceVolumeChange}
                      className="slider-input"
                    />
                  </div>

                  {/* Source Audio Waveform Box / Live Recording */}
                  {isRecording && recordingTarget === 1 ? (
                    <div className="waveform-box source-box" style={{ borderColor: 'rgba(255, 59, 48, 0.3)' }}>
                      <div className="waveform-meta-row">
                        <span className="waveform-label" style={{ color: 'var(--accent-red)' }}>
                          Recording Live Microphone Input...
                        </span>
                        <span className="recording-timer" style={{ color: 'var(--accent-red)' }}>{recordingTime}s</span>
                      </div>
                      
                      <canvas ref={canvasRef} className="live-recorder-canvas" width={800} height={80} />
                      
                      <div className="waveform-playback-controls">
                        <button 
                          className="btn-play-center"
                          style={{ borderColor: 'var(--accent-red)', color: 'var(--accent-red)', background: 'rgba(255, 59, 48, 0.05)' }}
                          onClick={stopRecording}
                          title="Stop Recording"
                        >
                          <Square size={16} fill="#ff3b30" />
                        </button>
                      </div>
                    </div>
                  ) : activeSourceUrl ? (
                    <div className="waveform-box source-box">
                      <div className="waveform-meta-row">
                        <span className="waveform-label">
                          <Music size={14} />
                          Source Audio Waveform
                        </span>
                        <span className="waveform-filename-text">
                          {initAudio ? initAudio.filename : 'source_audio.wav'}
                        </span>
                      </div>
                      
                      <div className="wavesurfer-wave" ref={sourceContainerRef}></div>
                      
                      {inpaintRange && (
                        <div className="region-info-overlay">
                          <span>Inpaint Start: {inpaintRange[0]}s</span>
                          <span>Inpaint End: {inpaintRange[1]}s</span>
                          <span>Duration: {(inpaintRange[1] - inpaintRange[0]).toFixed(2)}s</span>
                        </div>
                      )}

                      <div className="waveform-playback-controls">
                        <button 
                          className="btn-play-center"
                          onClick={handlePlayPauseSource}
                          title={isSourcePlaying ? 'Pause' : 'Play'}
                        >
                          {isSourcePlaying ? <Square size={16} fill="#fff" /> : <Play size={16} fill="#fff" />}
                        </button>

                        {inpaintRange && (
                          <button 
                            className="btn-clear-region"
                            onClick={handleClearSourceRegion}
                          >
                            Clear Region
                          </button>
                        )}
                      </div>
                    </div>
                  ) : (
                    /* Empty Symmetrical Upload Dropzone */
                    <div 
                      className={`waveform-box source-box ${dragOver ? 'drag-over' : ''}`}
                      style={{ borderStyle: 'dashed', minHeight: '150px', justifyContent: 'center', alignItems: 'center', cursor: 'pointer' }}
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                    >
                      <Upload size={24} style={{ color: 'var(--accent-cyan)', marginBottom: '0.5rem' }} />
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {audioUploadLoading ? 'Uploading...' : 'Drag & Drop source audio here'}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        or click to browse
                      </span>
                    </div>
                  )}

                  {/* Upload & Record Buttons */}
                  {!isRecording && (
                    <div className="media-actions" style={{ borderTop: 'none', paddingTop: 0, marginBottom: '1.2rem' }}>
                      <input 
                        type="file" 
                        id="audio-file" 
                        ref={fileInputRef}
                        onChange={(e) => handleFileChange(e, 1)}
                        accept="audio/*" 
                        style={{ display: 'none' }} 
                      />
                      <button 
                        className="upload-btn-label"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <Upload size={14} />
                        Upload Audio File
                      </button>
                      <button 
                        className="record-btn"
                        onClick={() => startRecording(1)}
                      >
                        <Mic size={14} />
                        Record Mic
                      </button>
                    </div>
                  )}
                </>
              ) : sourceMode === 'blend' ? (
                /* DUAL AUDIO BLENDING MODE INPUTS */
                <>
                  {/* SOURCE 1 BLOCK */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', width: '100%' }}>
                    <div className="volume-control-group">
                      <div className="volume-header">
                        <span style={{ fontSize: '0.7rem', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>SOURCE A VOLUME</span>
                        <span className="volume-value">{Math.round(sourceVolume * 100)}%</span>
                      </div>
                      <input 
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={sourceVolume}
                        onChange={handleSourceVolumeChange}
                        className="slider-input"
                      />
                    </div>

                    {isRecording && recordingTarget === 1 ? (
                      <div className="waveform-box source-box" style={{ borderColor: 'rgba(255, 59, 48, 0.3)', minHeight: '120px' }}>
                        <div className="waveform-meta-row">
                          <span className="waveform-label" style={{ color: 'var(--accent-red)', fontSize: '0.7rem' }}>
                            Recording A...
                          </span>
                          <span className="recording-timer" style={{ color: 'var(--accent-red)', fontSize: '0.7rem' }}>{recordingTime}s</span>
                        </div>
                        <canvas ref={canvasRef} className="live-recorder-canvas" width={800} height={40} style={{ height: '40px' }} />
                        <div className="waveform-playback-controls">
                          <button className="btn-play-center" onClick={stopRecording}><Square size={12} fill="#ff3b30" /></button>
                        </div>
                      </div>
                    ) : activeSourceUrl ? (
                      <div className="waveform-box source-box" style={{ padding: '1rem', gap: '0.8rem' }}>
                        <div className="waveform-meta-row">
                          <span className="waveform-label" style={{ fontSize: '0.7rem' }}><Music size={12} />SOURCE A</span>
                          <span className="waveform-filename-text" style={{ fontSize: '0.65rem' }}>{initAudio?.filename}</span>
                        </div>
                        <div className="wavesurfer-wave" ref={sourceContainerRef} style={{ height: '60px' }}></div>
                        <div className="waveform-playback-controls" style={{ marginTop: 0 }}>
                          <button className="btn-play-center" onClick={handlePlayPauseSource}>
                            {isSourcePlaying ? <Square size={12} fill="#fff" /> : <Play size={12} fill="#fff" />}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div 
                        className={`waveform-box source-box ${dragOver ? 'drag-over' : ''}`}
                        style={{ borderStyle: 'dashed', minHeight: '110px', justifyContent: 'center', alignItems: 'center', cursor: 'pointer', padding: '1rem' }}
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                      >
                        <Upload size={16} style={{ color: 'var(--accent-cyan)', marginBottom: '0.2rem' }} />
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                          {audioUploadLoading ? 'Uploading A...' : 'Upload Source A'}
                        </span>
                      </div>
                    )}

                    {!isRecording && (
                      <div className="media-actions" style={{ borderTop: 'none', paddingTop: 0, gap: '0.4rem', justifyContent: 'flex-start' }}>
                        <input type="file" ref={fileInputRef} onChange={(e) => handleFileChange(e, 1)} accept="audio/*" style={{ display: 'none' }} />
                        <button className="upload-btn-label" onClick={() => fileInputRef.current?.click()} style={{ padding: '0.3rem 0.6rem', fontSize: '0.65rem' }}><Upload size={10} />Upload A</button>
                        <button className="record-btn" onClick={() => startRecording(1)} style={{ padding: '0.3rem 0.6rem', fontSize: '0.65rem' }}><Mic size={10} />Record A</button>
                      </div>
                    )}
                  </div>

                  {/* SOURCE 2 BLOCK */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', width: '100%', marginTop: '0.5rem' }}>
                    <div className="volume-control-group">
                      <div className="volume-header">
                        <span style={{ fontSize: '0.7rem', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>SOURCE B VOLUME</span>
                        <span className="volume-value">{Math.round(sourceVolume2 * 100)}%</span>
                      </div>
                      <input 
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={sourceVolume2}
                        onChange={handleSourceVolumeChange2}
                        className="slider-input"
                      />
                    </div>

                    {isRecording && recordingTarget === 2 ? (
                      <div className="waveform-box source-box" style={{ borderColor: 'rgba(255, 59, 48, 0.3)', minHeight: '120px' }}>
                        <div className="waveform-meta-row">
                          <span className="waveform-label" style={{ color: 'var(--accent-red)', fontSize: '0.7rem' }}>
                            Recording B...
                          </span>
                          <span className="recording-timer" style={{ color: 'var(--accent-red)', fontSize: '0.7rem' }}>{recordingTime}s</span>
                        </div>
                        <canvas ref={canvasRef} className="live-recorder-canvas" width={800} height={40} style={{ height: '40px' }} />
                        <div className="waveform-playback-controls">
                          <button className="btn-play-center" onClick={stopRecording}><Square size={12} fill="#ff3b30" /></button>
                        </div>
                      </div>
                    ) : activeSourceUrl2 ? (
                      <div className="waveform-box source-box" style={{ padding: '1rem', gap: '0.8rem' }}>
                        <div className="waveform-meta-row">
                          <span className="waveform-label" style={{ fontSize: '0.7rem' }}><Music size={12} />SOURCE B</span>
                          <span className="waveform-filename-text" style={{ fontSize: '0.65rem' }}>{initAudio2?.filename}</span>
                        </div>
                        <div className="wavesurfer-wave" ref={sourceContainerRef2} style={{ height: '60px' }}></div>
                        <div className="waveform-playback-controls" style={{ marginTop: 0 }}>
                          <button className="btn-play-center" onClick={handlePlayPauseSource2}>
                            {isSourcePlaying2 ? <Square size={12} fill="#fff" /> : <Play size={12} fill="#fff" />}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div 
                        className={`waveform-box source-box ${dragOver2 ? 'drag-over' : ''}`}
                        style={{ borderStyle: 'dashed', minHeight: '110px', justifyContent: 'center', alignItems: 'center', cursor: 'pointer', padding: '1rem' }}
                        onClick={() => fileInputRef2.current?.click()}
                        onDragOver={handleDragOver2}
                        onDragLeave={handleDragLeave2}
                        onDrop={handleDrop2}
                      >
                        <Upload size={16} style={{ color: 'var(--accent-cyan)', marginBottom: '0.2rem' }} />
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                          {audioUploadLoading2 ? 'Uploading B...' : 'Upload Source B'}
                        </span>
                      </div>
                    )}

                    {!isRecording && (
                      <div className="media-actions" style={{ borderTop: 'none', paddingTop: 0, gap: '0.4rem', justifyContent: 'flex-start', marginBottom: '0.8rem' }}>
                        <input type="file" ref={fileInputRef2} onChange={(e) => handleFileChange(e, 2)} accept="audio/*" style={{ display: 'none' }} />
                        <button className="upload-btn-label" onClick={() => fileInputRef2.current?.click()} style={{ padding: '0.3rem 0.6rem', fontSize: '0.65rem' }}><Upload size={10} />Upload B</button>
                        <button className="record-btn" onClick={() => startRecording(2)} style={{ padding: '0.3rem 0.6rem', fontSize: '0.65rem' }}><Mic size={10} />Record B</button>
                      </div>
                    )}
                  </div>

                  {/* BLENDING SETTINGS BOX */}
                  <div className="parameters-box" style={{ padding: '1rem', gap: '0.8rem', marginBottom: '1.2rem' }}>
                    <div className="parameter-group">
                      <div className="parameter-header">
                        <span className="parameter-label" style={{ fontSize: '0.7rem' }}>BLEND RATIO (A VS B)</span>
                        <span className="parameter-value" style={{ fontSize: '0.75rem' }}>{Math.round((1 - blendRatio) * 100)}% A / {Math.round(blendRatio * 100)}% B</span>
                      </div>
                      <input 
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={blendRatio}
                        onChange={(e) => setBlendRatio(Number(e.target.value))}
                        className="slider-input"
                      />
                    </div>

                    <div className="parameter-group">
                      <div className="parameter-header">
                        <span className="parameter-label" style={{ fontSize: '0.7rem' }}>BLENDING METHOD EXPERIMENT</span>
                      </div>
                      <select 
                        className="model-select-dropdown" 
                        value={blendMethod} 
                        onChange={(e) => setBlendMethod(e.target.value as any)}
                        style={{ padding: '0.4rem', fontSize: '0.75rem', width: '100%' }}
                      >
                        <option value="waveform">A. Pre-Encoder Waveform Mix (Time Domain)</option>
                        <option value="latent_linear">B. Post-Encoder Latent Linear Interpolation (Feature Morph)</option>
                        <option value="latent_slerp">C. Spherical Latent Interpolation (Constant Energy Slerp) (TODO)</option>
                        <option value="asymmetric_noise">D. Asymmetric Noising (Structural Mold) (TODO)</option>
                        <option value="vector_field">E. Vector Field Interpolation (Dynamic compromise) (TODO)</option>
                      </select>
                    </div>
                  </div>
                </>
              ) : (
                /* LIVE SOURCE (MIC / INPUT) MODE */
                <>
                  {/* DEVICE & CHANNEL CONFIGURATION */}
                  <div className="parameters-box" style={{ padding: '1rem', gap: '0.8rem', marginBottom: '0.8rem' }}>
                    <div style={{ display: 'flex', gap: '0.8rem', width: '100%', flexWrap: 'wrap' }}>
                      
                      {/* Hardware Input Device Selector */}
                      <div className="parameter-group" style={{ flex: '1 1 200px' }}>
                        <div className="parameter-header">
                          <span className="parameter-label" style={{ fontSize: '0.7rem' }}>AUDIO INPUT DEVICE</span>
                          <button 
                            onClick={refreshAudioDevices}
                            style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.2rem', fontSize: '0.65rem' }}
                            title="Refresh connected devices"
                          >
                            <RefreshCw size={10} /> Scan
                          </button>
                        </div>
                        <select
                          className="model-select-dropdown"
                          value={selectedDeviceId}
                          onChange={(e) => {
                            setSelectedDeviceId(e.target.value);
                            if (isLiveFeedActive) {
                              setTimeout(() => startLiveFeed(), 50);
                            }
                          }}
                          style={{ padding: '0.4rem', fontSize: '0.75rem', width: '100%' }}
                        >
                          {audioInputDevices.length === 0 ? (
                            <option value="">Default Microphone</option>
                          ) : (
                            audioInputDevices.map((d, idx) => (
                              <option key={d.deviceId || idx} value={d.deviceId}>
                                {d.label || `Audio Input ${idx + 1}`}
                              </option>
                            ))
                          )}
                        </select>
                      </div>

                      {/* Input Channel Routing */}
                      <div className="parameter-group" style={{ flex: '1 1 140px' }}>
                        <div className="parameter-header">
                          <span className="parameter-label" style={{ fontSize: '0.7rem' }}>CHANNEL ROUTING</span>
                        </div>
                        <select
                          className="model-select-dropdown"
                          value={selectedChannel}
                          onChange={(e) => setSelectedChannel(e.target.value as any)}
                          style={{ padding: '0.4rem', fontSize: '0.75rem', width: '100%' }}
                        >
                          <option value="stereo">Stereo (Both Channels)</option>
                          <option value="ch1">Channel 1 (Left / Mono)</option>
                          <option value="ch2">Channel 2 (Right)</option>
                        </select>
                      </div>
                    </div>

                    {/* Monitor Volume Slider */}
                    <div className="volume-control-group" style={{ marginTop: '0.2rem' }}>
                      <div className="volume-header">
                        <span style={{ fontSize: '0.7rem', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>
                          LIVE MONITOR VOLUME {liveMonitorVolume === 0 ? '(MUTED)' : ''}
                        </span>
                        <span className="volume-value" style={{ color: liveMonitorVolume > 0 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>
                          {Math.round(liveMonitorVolume * 100)}%
                        </span>
                      </div>
                      <input 
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={liveMonitorVolume}
                        onChange={handleLiveMonitorVolumeChange}
                        className="slider-input"
                      />
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.1rem' }}>
                        ⚠️ Keep at 0% when using speakers to prevent acoustic feedback loop.
                      </span>
                    </div>
                  </div>

                  {/* LIVE FEED ACTION & STATUS BAR */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem', gap: '0.6rem' }}>
                    <button
                      className="live-feed-btn"
                      onClick={isLiveFeedActive ? stopLiveFeed : startLiveFeed}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.55rem 1.1rem',
                        borderRadius: '8px',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        background: isLiveFeedActive ? 'rgba(255, 59, 48, 0.15)' : 'rgba(0, 240, 255, 0.12)',
                        color: isLiveFeedActive ? '#ff3b30' : '#00f0ff',
                        border: `1.5px solid ${isLiveFeedActive ? '#ff3b30' : '#00f0ff'}`
                      }}
                    >
                      {isLiveFeedActive ? (
                        <>
                          <Square size={13} fill="#ff3b30" />
                          <span>STOP LIVE FEED</span>
                        </>
                      ) : (
                        <>
                          <Mic size={13} />
                          <span>START LIVE FEED</span>
                        </>
                      )}
                    </button>

                    {/* Buffer Size Selector & Status Gauge */}
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.35rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <span style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', fontWeight: 800, letterSpacing: '0.05em' }}>BUFFER:</span>
                        <div style={{ display: 'flex', background: '#000000', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '6px', padding: '2px', gap: '2px' }}>
                          {[3, 10, 20].map((secs) => (
                            <button
                              key={secs}
                              onClick={() => setLiveBufferTargetSecs(secs)}
                              style={{
                                background: liveBufferTargetSecs === secs ? '#1e1e1e' : 'transparent',
                                border: liveBufferTargetSecs === secs ? '1px solid rgba(0, 240, 255, 0.3)' : '1px solid transparent',
                                color: liveBufferTargetSecs === secs ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                                borderRadius: '4px',
                                padding: '0.2rem 0.45rem',
                                fontSize: '0.62rem',
                                fontWeight: 800,
                                cursor: 'pointer',
                                transition: 'all 0.15s ease'
                              }}
                              title={`Set rolling buffer window to ${secs} seconds`}
                            >
                              {secs}s
                            </button>
                          ))}
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.68rem' }}>
                        <span style={{ 
                          display: 'inline-block', 
                          width: '7px', 
                          height: '7px', 
                          borderRadius: '50%', 
                          background: isLiveFeedActive ? '#00f0ff' : '#666',
                          boxShadow: isLiveFeedActive ? '0 0 8px #00f0ff' : 'none'
                        }}></span>
                        <span style={{ color: isLiveFeedActive ? 'var(--accent-cyan)' : 'var(--text-muted)', fontWeight: 700 }}>
                          {isLiveFeedActive ? 'LIVE ACTIVE' : 'STANDBY'}
                        </span>
                        <span style={{ fontSize: '0.65rem', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
                          ({Math.min(liveBufferTargetSecs, liveBufferDuration).toFixed(1)}s / {liveBufferTargetSecs}.0s)
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* LIVE OSCILLOSCOPE & BUFFER VISUALIZER */}
                  <div className="waveform-box source-box" style={{ padding: '1rem', gap: '0.6rem', position: 'relative' }}>
                    <div className="waveform-meta-row">
                      <span className="waveform-label" style={{ fontSize: '0.7rem', color: isLiveFeedActive ? '#00f0ff' : 'var(--text-muted)' }}>
                        <Activity size={12} />
                        {isLiveFeedActive ? `ROLLING ${liveBufferTargetSecs}-SECOND AUDIO BUFFER` : 'LIVE FEED STOPPED'}
                      </span>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {liveBufferDuration >= liveBufferTargetSecs 
                          ? `BUFFER READY (${liveBufferTargetSecs}.0s)` 
                          : `BUFFERING: ${Math.round((Math.min(liveBufferTargetSecs, liveBufferDuration) / liveBufferTargetSecs) * 100)}%`}
                      </span>
                    </div>

                    <canvas 
                      ref={liveCanvasRef} 
                      style={{ 
                        width: '100%', 
                        height: '85px', 
                        borderRadius: '6px', 
                        background: '#060606', 
                        display: 'block' 
                      }} 
                    />

                    {/* Input Peak VU Meter Bar */}
                    <div style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
                      <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', width: '30px' }}>PEAK</span>
                      <div style={{ flex: 1, height: '4px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            height: '100%', 
                            width: `${Math.min(100, livePeakLevel * 100)}%`, 
                            background: livePeakLevel > 0.85 ? '#ff3b30' : livePeakLevel > 0.6 ? '#ffd60a' : '#00f0ff',
                            transition: 'width 0.05s ease'
                          }} 
                        />
                      </div>
                      <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', width: '35px', textAlign: 'right' }}>
                        {Math.round(livePeakLevel * 100)}%
                      </span>
                    </div>

                    {/* Helper caption */}
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '0.2rem' }}>
                      💡 Click <strong>GENERATE</strong> at any time to freeze and synthesize the rolling {liveBufferTargetSecs}-second buffer.
                    </div>
                  </div>
                </>
              )}

            </div>

            {/* RIGHT COLUMN: GENERATED AUDIO (OUTPUT) & NOISE VARS */}
            <div className="split-column right">
              
              {/* Generated Volume Slider */}
              <div className="volume-control-group">
                <div className="volume-header">
                  <span>Generated Volume</span>
                  <span className="volume-value">{Math.round(outputVolume * 100)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.05"
                  value={outputVolume} 
                  onChange={handleOutputVolumeChange}
                  className="slider-input"
                />
              </div>

              {/* Generated Audio Waveform Box & Sidecar 3D Variant Scroll Wheel */}
              {activeOutputUrl ? (
                <div style={{ display: 'flex', alignItems: 'stretch', gap: '1.2rem', width: '100%' }}>
                  
                  {/* Primary Waveform Box */}
                  <div className="waveform-box output-box" style={{ flex: 1 }}>
                    <div className="waveform-meta-row">
                      <span className="waveform-label">
                        <Sparkles size={14} />
                        Generated Audio Waveform
                      </span>
                      <span className="waveform-filename-text">
                        {generatedAudio ? generatedAudio.filename : 'output.wav'}
                      </span>
                    </div>
                    
                    <div className="wavesurfer-wave" ref={outputContainerRef}></div>

                    <div className="waveform-playback-controls">
                      <button 
                        className="btn-play-center"
                        onClick={handlePlayPauseOutput}
                        title={isOutputPlaying ? 'Pause' : 'Play'}
                      >
                        {isOutputPlaying ? <Square size={16} fill="#fff" /> : <Play size={16} fill="#fff" />}
                      </button>

                      {/* Loop Button for Output Audio (Icon-only) */}
                      <button 
                        onClick={() => {
                          const next = !isOutputLooping;
                          setIsOutputLooping(next);
                          isOutputLoopingRef.current = next;
                        }}
                        title={isOutputLooping ? 'Disable Loop' : 'Enable Loop'}
                        style={{
                          width: '36px',
                          height: '36px',
                          borderRadius: '8px',
                          border: isOutputLooping ? '1.5px solid var(--accent-green)' : '1px solid rgba(255, 255, 255, 0.1)',
                          background: isOutputLooping ? 'rgba(0, 255, 102, 0.12)' : 'rgba(255, 255, 255, 0.03)',
                          color: isOutputLooping ? 'var(--accent-green)' : 'var(--text-secondary)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          boxShadow: isOutputLooping ? '0 0 10px rgba(0, 255, 102, 0.2)' : 'none'
                        }}
                      >
                        <Repeat size={15} />
                      </button>

                      <button 
                        className="btn-promote-source"
                        onClick={handlePromoteToSource}
                        title="Set this output as the new source audio to iterate further"
                      >
                        Use as Source
                        <Plus size={14} />
                      </button>
                    </div>
                  </div>

                  {/* 3D Cylindrical Variant Scroll Wheel (Cylinder Dial) with Symmetrical Nav Buttons */}
                  {variants.length > 0 && activeTab === 'generate_a' && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem' }}>
                      
                      {/* Symmetrical Up Button next to Wheel */}
                      <button 
                        className="wheel-nav-btn"
                        onClick={handleWheelUp}
                        disabled={variants.length === 0}
                        title="Move up the noise wheel (Less noise)"
                      >
                        <ChevronUp size={14} />
                      </button>

                      {/* Viewport */}
                      <div className="variant-wheel-viewport">
                        <div className="variant-wheel-title">NOISE VARS</div>
                        <div className="variant-wheel-track" style={{ transform: `translateY(${74 - activeVariantIndex * 32}px)` }}>
                          {variants.map((v, idx) => {
                            const isBase = idx === 0;
                            const distance = Math.abs(idx - activeVariantIndex);
                            let distClass = "dist-3";
                            if (distance === 0) distClass = "dist-0";
                            else if (distance === 1) distClass = "dist-1";
                            else if (distance === 2) distClass = "dist-2";

                            const hasAnyReady = v.seeds.some(s => s.isReady);

                            return (
                              <button
                                key={idx}
                                className={`variant-node ${hasAnyReady ? 'ready' : ''} ${idx === activeVariantIndex ? 'active' : ''} ${distClass}`}
                                disabled={!hasAnyReady}
                                onClick={() => loadVariant2D(idx, activeSeedIndex)}
                                title={`${v.delta > 0 ? '+' : ''}${v.delta.toFixed(2)} Delta (${v.noise_level.toFixed(2)} Noise)`}
                              >
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px', width: '100%' }}>
                                  <span style={{ fontSize: '0.58rem' }}>
                                    {isBase ? 'BASE' : `+${v.delta.toFixed(2)}`}
                                  </span>
                                  <div className="seed-dots-row">
                                    {v.seeds.map((s, sIdx) => (
                                      <span 
                                        key={sIdx} 
                                        className={`seed-dot ${s.isReady ? 'ready' : ''} ${sIdx === activeSeedIndex && idx === activeVariantIndex ? 'active' : ''}`}
                                      />
                                    ))}
                                  </div>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                        {/* Bezel bracket highlight */}
                        <div className="wheel-center-bracket"></div>
                      </div>

                      {/* Symmetrical Down Button next to Wheel */}
                      <button 
                        className="wheel-nav-btn"
                        onClick={handleWheelDown}
                        disabled={variants.length === 0}
                        title="Move down the noise wheel (More noise)"
                      >
                        <ChevronDown size={14} />
                      </button>
                    </div>
                  )}

                </div>
              ) : (
                /* Empty Symmetrical Output Placeholder */
                <div 
                  className="waveform-box output-box"
                  style={{ borderStyle: 'dashed', minHeight: '180px', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}
                >
                  <Sparkles size={24} style={{ marginBottom: '0.5rem' }} />
                  <span style={{ fontSize: '0.85rem' }}>
                    Generated audio will appear here
                  </span>
                </div>
              )}

              {/* 2D GENERATION MATRIX MONITOR PANEL */}
              <div className="matrix-monitor-panel" style={{ marginTop: '0.5rem' }}>
                <div className="monitor-modal-header" style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '0.6rem', marginBottom: '0.6rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                    <Grid size={16} className="icon-glow-cyan" />
                    <span className="monitor-modal-title" style={{ fontSize: '0.75rem' }}>2D GENERATION MATRIX MONITOR</span>
                  </div>
                </div>

                {variants.length === 0 ? (
                  <div className="waveform-box" style={{ borderStyle: 'dashed', minHeight: '120px', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)', flex: 1, gap: '0.4rem' }}>
                    <Grid size={20} style={{ color: 'var(--text-muted)' }} />
                    <span style={{ fontSize: '0.8rem' }}>Matrix Monitor offline</span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Generate audio in Audio-to-Audio mode to see the variations.</span>
                  </div>
                ) : (
                  <>
                    <div className="monitor-modal-meta" style={{ gap: '1rem', padding: '0.4rem 0.8rem', marginBottom: '0.8rem' }}>
                      <span className="meta-item"><strong>PROMPT:</strong> "{prompt || 'Unconditional'}"</span>
                      <span className="meta-item"><strong>BASE SEED:</strong> {variants[0]?.seeds[0]?.seed || 'N/A'}</span>
                    </div>

                    <div className="matrix-grid-with-scrubber">
                      {/* Vertical Joystick Scrubber on Left */}
                      <div className="matrix-vertical-slider-container">
                        <div className="vslider-header-badge">
                          <Sliders size={12} style={{ color: 'var(--accent-cyan)' }} />
                          <span className="vslider-text">
                            {variants[activeVariantIndex]?.delta === 0 ? 'BASE' : `+${variants[activeVariantIndex]?.delta.toFixed(2)}`}
                          </span>
                        </div>
                        
                        <div className="vslider-track-wrap">
                          <input 
                            type="range"
                            min="0"
                            max={variants.length - 1}
                            step="1"
                            value={(variants.length - 1) - activeVariantIndex}
                            onChange={(e) => {
                              const targetIndex = (variants.length - 1) - Number(e.target.value);
                              loadVariant2D(targetIndex, activeSeedIndex);
                            }}
                            className="matrix-vslider"
                            title="Vertical Noise Level Joystick: Drag smoothly to crossfade between variations"
                          />
                        </div>
                      </div>

                      {/* Columns & Rows Container */}
                      <div className="monitor-modal-grid-container" style={{ maxHeight: '350px' }}>
                        {/* Columns Header (S1..S5) */}
                        <div className="monitor-modal-cols-header" style={{ gap: '6px', marginBottom: '6px' }}>
                          <div style={{ width: '45px' }}></div>
                          {Array.from({ length: 5 }, (_, sIdx) => (
                            <div key={sIdx} className="col-modal-label" style={{ fontSize: '0.55rem' }}>VAR {sIdx + 1}</div>
                          ))}
                        </div>
                        
                        {/* Rows */}
                        <div className="monitor-modal-rows-container" style={{ gap: '4px' }}>
                          {variants.map((v, vIdx) => {
                            const isBase = vIdx === 0;
                            return (
                              <div key={vIdx} className="monitor-modal-row" style={{ gridTemplateColumns: '45px repeat(5, 1fr)', gap: '6px' }}>
                                <div className="row-modal-label" style={{ fontSize: '0.55rem', paddingRight: '6px' }}>{isBase ? 'BASE' : `+${v.delta.toFixed(2)}`}</div>
                                <div className="row-modal-cells">
                                  {v.seeds.map((s, sIdx) => {
                                    const isGeneratingCell = generatingCell && generatingCell.vIndex === vIdx && generatingCell.sIndex === sIdx;
                                    const isActive = vIdx === activeVariantIndex && sIdx === activeSeedIndex;
                                    
                                    let cellClass = "";
                                    if (isActive) cellClass = "active";
                                    else if (isGeneratingCell) cellClass = "generating";
                                    else if (s.isReady) cellClass = "ready";
                                    
                                    return (
                                      <button
                                        key={sIdx}
                                        className={`monitor-modal-cell ${cellClass}`}
                                        disabled={!s.isReady}
                                        onClick={() => loadVariant2D(vIdx, sIdx)}
                                        title={`Noise Level: ${v.noise_level.toFixed(2)} (${v.delta > 0 ? '+' : ''}${v.delta.toFixed(2)} Delta), Seed: ${s.seed || 'Pending'} (${s.isReady ? 'Ready - Click to audition' : isGeneratingCell ? 'Generating...' : 'In Queue'})`}
                                      />
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                    
                    {/* Legend */}
                    <div className="monitor-modal-legend" style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '0.6rem', marginTop: '0.6rem', flexWrap: 'wrap', gap: '0.8rem', fontSize: '0.65rem' }}>
                      <div className="legend-item"><span className="legend-modal-dot ready" style={{ width: '8px', height: '8px' }}></span>Ready</div>
                      <div className="legend-item"><span className="legend-modal-dot generating" style={{ width: '8px', height: '8px' }}></span>Rendering</div>
                      <div className="legend-item"><span className="legend-modal-dot active" style={{ width: '8px', height: '8px' }}></span>Playing</div>
                      <div className="legend-item"><span className="legend-modal-dot pending" style={{ width: '8px', height: '8px' }}></span>Queued</div>
                    </div>
                  </>
                )}
              </div>
            </div>

          </div>



        </div>

        {/* 4. BLURRED GLASS BACKDROP */}
        <div 
          className={`drawer-backdrop ${showHistory ? 'open' : ''}`}
          onClick={() => setShowHistory(false)}
        ></div>

        {/* 5. SLIDE-OUT DRAWER PANEL */}
        <aside className={`history-drawer ${showHistory ? 'open' : ''}`}>
          <div className="history-header">
            <span className="history-title">Creation History</span>
            <button 
              className="btn-close-drawer" 
              onClick={() => setShowHistory(false)}
              title="Close History"
            >
              <X size={16} />
            </button>
          </div>

          <div className="history-list">
            {history.length === 0 ? (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem', padding: '2rem' }}>
                No past generations found.<br/>Your creations will appear here.
              </div>
            ) : (
              history.map((item) => (
                <div 
                  key={item.id} 
                  className={`history-item ${activeSourceUrl?.includes(item.filename) ? 'active' : ''}`}
                  onClick={() => handleHistoryItemClick(item)}
                >
                  <div className="history-item-prompt">{item.prompt || 'Unconditional'}</div>
                  <div className="history-item-meta">
                    <span>{item.seconds}s • {item.steps} steps</span>
                    <span>{item.dit_name}</span>
                  </div>
                  <div className="history-item-actions">
                    <button 
                      className="history-action-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        const link = document.createElement('a');
                        link.href = `${API_URL}${item.wav_url}`;
                        link.download = item.filename;
                        link.click();
                      }}
                    >
                      <Download size={12} />
                      Download
                    </button>
                    <button 
                      className="history-action-btn delete"
                      onClick={(e) => handleDeleteHistory(e, item.id)}
                    >
                      <Trash2 size={12} />
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>


      </main>
    </div>
  );
}
