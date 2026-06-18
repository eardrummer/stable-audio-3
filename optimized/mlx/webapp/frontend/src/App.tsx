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
  Grid
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

export default function App() {
  // Unique connection ID for websocket routing
  const [connectionId] = useState(() => Math.random().toString(36).substring(2, 15));
  
  // Prompt States
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [useNegPrompt, setUseNegPrompt] = useState(false);
  
  // Audio Source (Input) States
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
  
  // Generated Audio (Output) States
  const [generatedAudio, setGeneratedAudio] = useState<{
    filename: string;
    duration: number;
    wav_url: string;
  } | null>(null);
  const [activeOutputUrl, setActiveOutputUrl] = useState<string | null>(null);
  const [isOutputPlaying, setIsOutputPlaying] = useState(false);
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
  const [audioUploadLoading, setAudioUploadLoading] = useState(false);
  
  // Generation Settings
  const [ditName, setDitName] = useState<'sm-music' | 'sm-sfx' | 'medium'>('sm-music');
  const [noiseLevel, setNoiseLevel] = useState(0.7); // 0.05 - 1.0 (for Generate A)
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
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Wavesurfer Refs
  const sourceWavesurferRef = useRef<WaveSurfer | null>(null);
  const sourceContainerRef = useRef<HTMLDivElement>(null);
  const sourceRegionsRef = useRef<any>(null);

  const outputWavesurferRef = useRef<WaveSurfer | null>(null);
  const outputContainerRef = useRef<HTMLDivElement>(null);

  // Playhead synchronization refs
  const pendingPlayheadRef = useRef<{ time: number; play: boolean } | null>(null);

  // Spectrogram Canvas Refs
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);

  // Web Audio Context & Analyser Refs (Source)
  const sourceAudioCtxRef = useRef<AudioContext | null>(null);
  const sourceAnalyserRef = useRef<AnalyserNode | null>(null);
  const connectedSourceMediaRef = useRef<HTMLMediaElement | null>(null);
  const sourceAnimationRef = useRef<number | null>(null);

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
          setActiveVariantIndex(10); // Reset to center
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

    ws.on('ready', () => {
      setIsOutputPlaying(false);
      setOutputProgress(0);

      // Synced Playhead Restoration!
      if (pendingPlayheadRef.current) {
        const { time, play } = pendingPlayheadRef.current;
        const duration = ws.getDuration();
        const seekTime = time < duration ? time : 0;
        ws.setTime(seekTime);
        
        if (play) {
          ws.play();
          setIsOutputPlaying(true);
          // If global synced play was active, resume source too
          if (isGlobalPlaying && sourceWavesurferRef.current) {
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
    });

    ws.on('finish', () => {
      if (isLoopingRef.current) {
        ws.play();
      } else {
        setIsOutputPlaying(false);
        setIsGlobalPlaying(false);
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

  const fetchModelStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/status`);
      const data = await res.json();
      setModelStatus(data.models);
    } catch (e) {
      console.error('Failed to fetch model status:', e);
    }
  };

  // Drag & Drop / File Upload Handlers
  const [dragOver, setDragOver] = useState(false);
  
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
      await uploadFile(file);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await uploadFile(file);
    }
  };

  const uploadFile = async (file: File) => {
    setAudioUploadLoading(true);
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
      setInitAudio({
        filename: data.filename,
        local_path: data.local_path,
        duration: data.duration,
        wav_url: `${API_URL}${data.wav_url}`
      });
      
      setActiveSourceUrl(`${API_URL}${data.wav_url}`);
      setInpaintRange(null);
    } catch (e: any) {
      setErrorMsg(e.message);
    } finally {
      setAudioUploadLoading(false);
    }
  };

  // Mic Recording
  const startRecording = async () => {
    setErrorMsg(null);
    audioChunksRef.current = [];
    setRecordingTime(0);
    
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
        const file = new File([audioBlob], 'mic_recording.webm', { type: 'audio/webm' });
        await uploadFile(file);
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

  // Real-time Mixer Volume Controls
  const handleSourceVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = Number(e.target.value);
    setSourceVolume(vol);
    if (sourceWavesurferRef.current) {
      sourceWavesurferRef.current.setVolume(vol * masterVolume);
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
    if (outputWavesurferRef.current) {
      outputWavesurferRef.current.setVolume(outputVolume * vol);
    }
  };

  // Synced Play/Pause Control
  const handleGlobalPlayPause = () => {
    const newState = !isGlobalPlaying;
    setIsGlobalPlaying(newState);
    
    const sourceWS = sourceWavesurferRef.current;
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
      }
      setIsSourcePlaying(true);
      setIsOutputPlaying(true);
    } else {
      if (sourceWS) sourceWS.pause();
      if (outputWS) outputWS.pause();
      setIsSourcePlaying(false);
      setIsOutputPlaying(false);
    }
  };

  // Sync Playheads & Return to Zero (Rewind to Start)
  const handleSyncPlayheads = () => {
    const sourceWS = sourceWavesurferRef.current;
    const outputWS = outputWavesurferRef.current;
    
    if (sourceWS && outputWS) {
      sourceWS.setTime(0);
      outputWS.setTime(0);
      
      // If global sync is rolling, keep them playing together from 0
      if (isGlobalPlaying) {
        sourceWS.play();
        outputWS.play();
      }
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
        
        // Save playhead & play state for restoration in 'ready'
        pendingPlayheadRef.current = { time: currentTime, play: wasPlaying };
        
        // Pause old player
        oldWS.pause();
        
        // Load new URL
        setActiveOutputUrl(seedVar.wav_url);
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
    if (!prompt.trim() && !initAudio) {
      setErrorMsg('Please enter a prompt or load an initial audio file.');
      return;
    }
    
    setIsGenerating(true);
    setErrorMsg(null);
    setProgressPercent(0);
    setProgressStep(0);
    setProgressTotal(steps);
    setProgressText('Warming up MLX models in unified VRAM...');

    // Initialize the 21 vertical noise levels, each with 5 horizontal seed variations
    const deltas = [-0.20, -0.18, -0.16, -0.14, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20];
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
    setActiveVariantIndex(10); // Center is index 10
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

  const handlePlayPauseOutput = () => {
    if (outputWavesurferRef.current) {
      outputWavesurferRef.current.playPause();
      setIsOutputPlaying(!isOutputPlaying);
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
      const deltas = [-0.20, -0.18, -0.16, -0.14, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20];
      
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
      setActiveVariantIndex(10); // Center
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

          {/* Grid Monitor Toggle Button */}
          <button 
            className={`header-monitor-btn ${showMonitor ? 'active' : ''}`}
            onClick={() => setShowMonitor(!showMonitor)}
            title="Open 2D Matrix Generation Monitor"
          >
            <Grid size={15} />
            Monitor
          </button>

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
          
          {/* TOP SPLIT: EXACTLY TWO COLUMNS WITH A LINE IN THE MIDDLE */}
          <div className="top-split-container">
            
            {/* LEFT COLUMN: AUDIO SOURCE (INPUT) */}
            <div className="split-column left">
              
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
              {isRecording ? (
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
                  style={{ borderStyle: 'dashed', minHeight: '180px', justifyContent: 'center', alignItems: 'center', cursor: 'pointer' }}
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
                <div className="media-actions" style={{ borderTop: 'none', paddingTop: 0 }}>
                  <input 
                    type="file" 
                    id="audio-file" 
                    ref={fileInputRef}
                    onChange={handleFileChange}
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
                    onClick={startRecording}
                  >
                    <Mic size={14} />
                    Record Mic
                  </button>
                </div>
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
                            const isCenter = idx === 10;
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
                                    {isCenter ? 'CENTER' : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(2)}`}
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
            </div>

          </div>

          {/* 3. BOTTOM CENTER PANEL: PROMPT & PARAMETERS */}
          <div className="bottom-center-panel">
            {isGenerating ? (
              /* PROGRESS OVERLAY */
              <div className="progress-overlay">
                <div className="progress-loader"></div>
                <div className="progress-text">{progressText}</div>
                <div className="progress-bar-container">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${progressPercent}%` }}
                  ></div>
                </div>
                <div className="progress-subtext">
                  Processing on Metal GPU (Unified VRAM)...
                </div>
              </div>
            ) : (
              <>
                {/* Centered Text Prompt Box */}
                <div className="prompt-input-container">
                  <div className="prompt-label">TEXT PROMPT</div>
                  <textarea
                    className="prompt-textarea"
                    placeholder="Enter a prompt to guide the generation..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </div>

                {/* Stable Audio Parameters Box */}
                <div className="parameters-box">
                  <div className="parameters-box-header-tabs">
                    <button 
                      className={`model-tab-btn ${activeTab === 'generate_a' ? 'active' : ''}`}
                      onClick={() => {
                        setActiveTab('generate_a');
                        handleClearSourceRegion();
                      }}
                    >
                      Audio-to-Audio / Inpaint
                    </button>
                    <button 
                      className={`model-tab-btn ${activeTab === 'generate_b' ? 'active' : ''}`}
                      onClick={() => {
                        setActiveTab('generate_b');
                        handleClearSourceRegion();
                      }}
                    >
                      Extend Audio
                    </button>
                  </div>

                  <div className="parameters-grid">
                    {/* Mode-specific Slider */}
                    {activeTab === 'generate_a' ? (
                      <div className="parameter-group">
                        <div className="parameter-header">
                          <span className="parameter-label">Adding Noise / Variation Strength</span>
                          <span className="parameter-value">{Math.round(noiseLevel * 100)}%</span>
                        </div>
                        <input 
                          type="range" 
                          min="0.05" 
                          max="1.0" 
                          step="0.05"
                          value={noiseLevel} 
                          onChange={(e) => setNoiseLevel(Number(e.target.value))}
                          className="slider-input"
                        />
                      </div>
                    ) : (
                      <div className="parameter-group">
                        <div className="parameter-header">
                          <span className="parameter-label">Extension Duration</span>
                          <span className="parameter-value">{extendLength} seconds</span>
                        </div>
                        <input 
                          type="range" 
                          min="5" 
                          max="30" 
                          step="1"
                          value={extendLength} 
                          onChange={(e) => setExtendLength(Number(e.target.value))}
                          className="slider-input"
                        />
                      </div>
                    )}

                    {/* CFG Slider */}
                    <div className="parameter-group">
                      <div className="parameter-header">
                        <span className="parameter-label">Control Strength (CFG)</span>
                        <span className="parameter-value">{cfg}</span>
                      </div>
                      <input 
                        type="range" 
                        min="1.0" 
                        max="12.0" 
                        step="0.5"
                        value={cfg} 
                        onChange={(e) => setCfg(Number(e.target.value))}
                        className="slider-input"
                      />
                    </div>

                    {/* Steps Slider */}
                    <div className="parameter-group">
                      <div className="parameter-header">
                        <span className="parameter-label">Denoising Steps</span>
                        <span className="parameter-value">{steps}</span>
                      </div>
                      <input 
                        type="range" 
                        min="1" 
                        max="32" 
                        step="1"
                        value={steps} 
                        onChange={(e) => setSteps(Number(e.target.value))}
                        className="slider-input"
                      />
                    </div>
                  </div>

                  {/* Advanced Options Toggle */}
                  <div style={{ alignSelf: 'flex-start' }}>
                    <button 
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem', cursor: 'pointer' }}
                    >
                      {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      {showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options'}
                    </button>
                  </div>

                  {showAdvanced && (
                    <div className="parameters-grid" style={{ gap: '1.2rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.2rem' }}>
                      <div className="parameter-group">
                        <div className="parameter-header">
                          <span className="parameter-label">Manual Seed (Optional)</span>
                        </div>
                        <input 
                          type="number"
                          placeholder="Random seed"
                          className="numeric-input"
                          style={{ width: '100%', textAlign: 'left' }}
                          value={seed}
                          onChange={(e) => setSeed(e.target.value === '' ? '' : Number(e.target.value))}
                        />
                      </div>
                      
                      <div className="parameter-group">
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.8rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                          <input 
                            type="checkbox" 
                            checked={freeModels}
                            onChange={(e) => setFreeModels(e.target.checked)}
                            style={{ accentColor: 'var(--accent-cyan)' }}
                          />
                          Free VRAM models immediately (recommended for 8GB Macs)
                        </label>
                      </div>
                    </div>
                  )}
                </div>

                {/* ERROR DISPLAY */}
                {errorMsg && (
                  <div style={{ color: 'var(--accent-red)', fontSize: '0.85rem', padding: '0.8rem', background: 'rgba(255, 59, 48, 0.03)', border: '1px solid rgba(255, 59, 48, 0.1)', borderRadius: '8px', width: '100%' }}>
                    {errorMsg}
                  </div>
                )}

                {/* GENERATE BUTTON */}
                <button 
                  className="btn-generate"
                  onClick={handleGenerate}
                  disabled={isGenerating || audioUploadLoading}
                >
                  <Sparkles size={16} />
                  {activeTab === 'generate_a' 
                    ? (inpaintRange ? 'Generate A: Inpaint Region' : 'Generate A: Audio-to-Audio') 
                    : 'Generate B: Extend Audio'}
                </button>
              </>
            )}
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

        {/* 6. 2D MATRIX MONITOR MODAL & BACKDROP (Opens floating on top of everything!) */}
        <div 
          className={`monitor-backdrop ${showMonitor ? 'open' : ''}`}
          onClick={() => setShowMonitor(false)}
        >
          <div 
            className="matrix-monitor-modal"
            onClick={(e) => e.stopPropagation()} // Prevent close on modal content click
          >
            <div className="monitor-modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                <Grid size={18} className="icon-glow-cyan" />
                <span className="monitor-modal-title">2D GENERATION MATRIX MONITOR</span>
              </div>
              <button 
                className="btn-close-monitor-modal" 
                onClick={() => setShowMonitor(false)}
                title="Close Monitor"
              >
                <X size={18} />
              </button>
            </div>
            
            <div className="monitor-modal-meta">
              <span className="meta-item"><strong>PROMPT:</strong> "{prompt || 'Unconditional'}"</span>
              <span className="meta-item"><strong>BASE SEED:</strong> {variants[10]?.seeds[0]?.seed || 'N/A'}</span>
            </div>

            <div className="monitor-modal-grid-container">
              {/* Columns Header (S1..S5) */}
              <div className="monitor-modal-cols-header">
                <div style={{ width: '60px' }}></div>
                {Array.from({ length: 5 }, (_, sIdx) => (
                  <div key={sIdx} className="col-modal-label">VARIATION {sIdx + 1}</div>
                ))}
              </div>
              
              {/* Rows */}
              <div className="monitor-modal-rows-container">
                {variants.map((v, vIdx) => {
                  const isCenter = vIdx === 10;
                  return (
                    <div key={vIdx} className="monitor-modal-row">
                      <div className="row-modal-label">{isCenter ? 'CTR' : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(2)}`}</div>
                      <div className="row-modal-cells">
                        {v.seeds.map((s, sIdx) => {
                          const isGenerating = generatingCell && generatingCell.vIndex === vIdx && generatingCell.sIndex === sIdx;
                          const isActive = vIdx === activeVariantIndex && sIdx === activeSeedIndex;
                          
                          let cellClass = "";
                          if (isActive) cellClass = "active";
                          else if (isGenerating) cellClass = "generating";
                          else if (s.isReady) cellClass = "ready";
                          
                          return (
                            <button
                              key={sIdx}
                              className={`monitor-modal-cell ${cellClass}`}
                              disabled={!s.isReady}
                              onClick={() => loadVariant2D(vIdx, sIdx)}
                              title={`Noise Level: ${v.noise_level.toFixed(2)} (${v.delta > 0 ? '+' : ''}${v.delta.toFixed(2)} Delta), Seed: ${s.seed || 'Pending'} (${s.isReady ? 'Ready - Click to audition' : isGenerating ? 'Generating...' : 'In Queue'})`}
                            />
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Legend */}
            <div className="monitor-modal-legend">
              <div className="legend-item"><span className="legend-modal-dot ready"></span>Ready to Audition</div>
              <div className="legend-item"><span className="legend-modal-dot generating"></span>Metal GPU Rendering</div>
              <div className="legend-item"><span className="legend-modal-dot active"></span>Currently Playing</div>
              <div className="legend-item"><span className="legend-modal-dot pending"></span>Queued / Pending</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
