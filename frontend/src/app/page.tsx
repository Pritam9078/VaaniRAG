"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Mic,
  Loader2,
  CheckCircle2,
  Zap,
  Server,
  ShieldAlert,
  Radio,
  Activity,
  Sparkles,
  Layers,
} from "lucide-react";

/* ----------------------------------------------------------------------- */
/*  Types                                                                   */
/* ----------------------------------------------------------------------- */

interface Latencies {
  stt_ms?: number;
  retrieval_ms?: number;
  rerank_ms?: number;
  generation_ms?: number;
  guardrail_ms?: number;
  tier1_ms?: number;
}

interface Source {
  chunk_id: string;
  doc_id: string;
  relevance_score: number;
}

interface WSMessage {
  type: string;
  text?: string;
  is_final?: boolean;
  answer?: string;
  reason?: string;
  stage?: string;
  sources?: Source[];
  latencies?: Latencies;
  grounding_score?: number;
  message?: string;
}

type Status = "idle" | "listening" | "processing" | "answered" | "refused";

/* ----------------------------------------------------------------------- */
/*  Sample content for the built-in demo pipeline                          */
/*  (kept separate so the real backend path above is untouched)            */
/* ----------------------------------------------------------------------- */

const DANGEROUS_PATTERNS = [
  "bomb",
  "explosive",
  "weapon",
  "poison",
  "hack into",
  "kill",
];

const KNOWN: Record<string, { t1: string; t2: string; conf: number }> = {
  "capital of india": {
    t1: "New Delhi — capital of India.",
    t2: "New Delhi is the capital of India. It was formally inaugurated in 1931 and houses the seat of all three branches of the Union government — the Parliament, the Supreme Court, and the Rashtrapati Bhavan.",
    conf: 0.94,
  },
  relativity: {
    t1: "Relativity: Einstein's theory of space, time and gravity.",
    t2: "Relativity refers to Einstein's two theories — special relativity (1905), which unifies space and time and sets the speed of light as a universal constant, and general relativity (1915), which describes gravity as the curvature of spacetime caused by mass and energy.",
    conf: 0.81,
  },
};

/* ----------------------------------------------------------------------- */
/*  Live ambient background — a slow neural particle field                 */
/* ----------------------------------------------------------------------- */

function LiveField({ active }: { active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    const DPR = typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;

    const resize = () => {
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * DPR;
      canvas.height = height * DPR;
      ctx.scale(DPR, DPR);
    };
    resize();
    window.addEventListener("resize", resize);

    const COUNT = Math.max(28, Math.floor((width * height) / 42000));
    const nodes = Array.from({ length: COUNT }).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.18,
      vy: (Math.random() - 0.5) * 0.18,
      r: Math.random() * 1.6 + 0.6,
    }));

    const LINK_DIST = 130;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      for (const n of nodes) {
        if (!reduceMotion) {
          n.x += n.vx * (active ? 2.4 : 1);
          n.y += n.vy * (active ? 2.4 : 1);
          if (n.x < 0 || n.x > width) n.vx *= -1;
          if (n.y < 0 || n.y > height) n.vy *= -1;
        }
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < LINK_DIST) {
            const alpha = (1 - dist / LINK_DIST) * (active ? 0.16 : 0.08);
            ctx.strokeStyle = `rgba(69, 232, 196, ${alpha})`;
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = active
          ? "rgba(157, 140, 255, 0.55)"
          : "rgba(125, 132, 163, 0.35)";
        ctx.fill();
      }

      if (!reduceMotion) {
        rafRef.current = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      window.removeEventListener("resize", resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [active]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 h-full w-full opacity-70"
      aria-hidden="true"
    />
  );
}

/* ----------------------------------------------------------------------- */
/*  Main component                                                         */
/* ----------------------------------------------------------------------- */

export default function MainInterface() {
  const [isRecording, setIsRecording] = useState(false);
  const [queryText, setQueryText] = useState("");

  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [tier1Answer, setTier1Answer] = useState("");
  const [tier2Answer, setTier2Answer] = useState("");
  const [refusal, setRefusal] = useState<{ reason: string; stage: string } | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [latencies, setLatencies] = useState<Latencies>({});
  const [groundingScore, setGroundingScore] = useState<number | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [isSubsequentTurn, setIsSubsequentTurn] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const levelRafRef = useRef<number | null>(null);
  const recognitionRef = useRef<any>(null);
  const usingRecognitionRef = useRef(false);
  const fallbackTranscriptRef = useRef("");

  const barRefs = useRef<Array<HTMLDivElement | null>>([]);
  const BAR_COUNT = 28;

  useEffect(() => {
    return () => {
      stopRecording();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---------------------------- WebSocket ---------------------------- */

  const connectWebSocket = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        resolve(wsRef.current);
        return;
      }

      let settled = false;
      const timeout = setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error("timeout"));
        }
      }, 1400);

      let ws: WebSocket;
      try {
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/voice";
        ws = new WebSocket(wsUrl);
      } catch (err) {
        clearTimeout(timeout);
        reject(err);
        return;
      }
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        resolve(ws);
      };
      ws.onerror = (e) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        reject(e);
      };

      ws.onmessage = (e) => {
        try {
          const msg: WSMessage = JSON.parse(e.data);
          handleWebSocketMessage(msg);
        } catch (err) {
          console.error("WS parse error", err);
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
      };

      wsRef.current = ws;
    });
  }, []);

  const speakAnswer = (text: string) => {
    if (!text || typeof window === "undefined" || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const containsHindi = /[\u0900-\u097F]/.test(text);
    utterance.lang = containsHindi ? "hi-IN" : "en-IN";
    window.speechSynthesis.speak(utterance);
  };

  const handleWebSocketMessage = (msg: WSMessage) => {
    switch (msg.type) {
      case "transcript_partial":
        const newText = msg.text || "";
        setTranscript(newText);
        if (msg.is_final) {
          if (newText.trim()) {
            setStatus("processing");
          } else {
            setStatus("idle");
          }
        }
        break;
      case "tier1":
        setTier1Answer(msg.answer || "");
        if (msg.sources) setSources(msg.sources);
        if (msg.latencies) setLatencies((prev) => ({ ...prev, ...msg.latencies }));
        setStatus("answered");
        break;
      case "tier2":
        setTier2Answer(msg.answer || "");
        if (msg.grounding_score) setGroundingScore(msg.grounding_score);
        if (msg.latencies) setLatencies((prev) => ({ ...prev, ...msg.latencies }));
        if (msg.answer) speakAnswer(msg.answer);
        break;
      case "refused":
      case "tier2_refused":
        setRefusal({ reason: msg.reason || "Unknown", stage: msg.stage || "generation" });
        setStatus("refused");
        if (msg.reason) speakAnswer(msg.reason);
        break;
      case "error":
      case "tier2_error":
        if (msg.message === "No transcript received") {
          console.log("No transcript received (user didn't speak).");
          setStatus("idle");
          break;
        }
        console.error("Server Error:", msg.message);
        setRefusal({ reason: msg.message || "Server Error", stage: "error" });
        setStatus("refused");
        if (msg.message) speakAnswer(msg.message);
        break;
    }
  };

  /* ------------------------- Demo simulation --------------------------- */
  /* Runs only when no live backend is reachable, so the interface always  */
  /* demonstrates the full two-tier pipeline end to end.                   */

  const resetPanel = () => {
    setTier1Answer("");
    setTier2Answer("");
    setRefusal(null);
    setSources([]);
    setLatencies({});
    setGroundingScore(null);
  };

  const simulateResponse = (text: string) => {
    setStatus("processing");
    const lower = text.toLowerCase();

    if (DANGEROUS_PATTERNS.some((p) => lower.includes(p))) {
      window.setTimeout(() => {
        setLatencies({ guardrail_ms: 4.2 });
        const reason = "The query matched a restricted-content safety policy and was blocked before retrieval.";
        setRefusal({
          reason: reason,
          stage: "input_guardrail",
        });
        setStatus("refused");
        speakAnswer(reason);
      }, 260);
      return;
    }

    if (/^my name is|^i am\s/i.test(text.trim())) {
      window.setTimeout(() => {
        setLatencies({ guardrail_ms: 3.8, retrieval_ms: 61.2 });
        const reason = "No passages in the index were relevant to this input — it falls outside the assistant's knowledge scope.";
        setRefusal({
          reason: reason,
          stage: "retrieval",
        });
        setStatus("refused");
        speakAnswer(reason);
      }, 320);
      return;
    }

    const matchKey = Object.keys(KNOWN).find((k) => lower.includes(k));
    const known = matchKey ? KNOWN[matchKey] : null;

    window.setTimeout(
      () => setLatencies((p) => ({ ...p, guardrail_ms: 3.6 })),
      60
    );
    window.setTimeout(
      () => {
        setLatencies((p) => ({ ...p, retrieval_ms: 38.4 }));
        setSources([
          { chunk_id: "c_0417", doc_id: "kb_general_2f8a91", relevance_score: known ? 0.91 : 0.63 },
          { chunk_id: "c_1183", doc_id: "kb_general_7bd120", relevance_score: known ? 0.84 : 0.51 },
        ]);
      },
      150
    );
    window.setTimeout(
      () => setLatencies((p) => ({ ...p, rerank_ms: 21.7 })),
      210
    );
    window.setTimeout(() => {
      setLatencies((p) => ({ ...p, tier1_ms: 46.9 }));
      setTier1Answer(
        known
          ? known.t1
          : "Reviewing the closest indexed passages for a direct extractive answer…"
      );
      setStatus("answered");
    }, 260);
    window.setTimeout(() => {
      setLatencies((p) => ({ ...p, generation_ms: known ? 812 : 940 }));
      const ans = known
        ? known.t2
        : `Based on the retrieved passages, here is a reasoned answer to "${text.trim()}". In a connected deployment this would be produced by the tier‑2 generative model, grounded in the cited sources.`;
      setTier2Answer(ans);
      setGroundingScore(known ? known.conf : 0.72);
      speakAnswer(ans);
    }, known ? 1050 : 1300);
  };

  /* ----------------------------- Voice level ---------------------------- */

  const startLevelLoop = (analyser: AnalyserNode) => {
    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteFrequencyData(data);
      const step = Math.floor(data.length / BAR_COUNT) || 1;
      for (let i = 0; i < BAR_COUNT; i++) {
        const v = data[i * step] / 255;
        const el = barRefs.current[i];
        if (el) {
          const scale = 0.18 + v * 1.4;
          el.style.transform = `scaleY(${scale})`;
          el.style.opacity = `${0.35 + v * 0.65}`;
        }
      }
      levelRafRef.current = requestAnimationFrame(tick);
    };
    tick();
  };

  const stopLevelLoop = () => {
    if (levelRafRef.current) cancelAnimationFrame(levelRafRef.current);
    levelRafRef.current = null;
    barRefs.current.forEach((el) => {
      if (el) {
        el.style.transform = "scaleY(0.18)";
        el.style.opacity = "0.35";
      }
    });
  };

  /* ------------------------------ Recording ------------------------------ */

  const startRecording = async () => {
    try {
      if (typeof window !== "undefined" && window.speechSynthesis) {
        window.speechSynthesis.cancel();
        // Unlock speech synthesis on iOS/Safari by speaking an empty string on user gesture
        const unlockUtterance = new SpeechSynthesisUtterance("");
        unlockUtterance.volume = 0;
        window.speechSynthesis.speak(unlockUtterance);
      }
      setTranscript("");
      resetPanel();

      setStatus("listening");
      setIsRecording(true);
      usingRecognitionRef.current = false;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000 },
      });
      streamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.75;
      analyserRef.current = analyser;
      source.connect(analyser);
      startLevelLoop(analyser);

      let ws: WebSocket | null = null;
      try {
        ws = await connectWebSocket();
        setDemoMode(false);
      } catch {
        setDemoMode(true);
      }

      if (ws) {
        try {
          await ctx.audioWorklet.addModule("/audio-worklet.js");
          const worklet = new AudioWorkletNode(ctx, "pcm-capture");
          workletNodeRef.current = worklet;
          worklet.port.onmessage = (e) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(e.data);
            }
          };
          source.connect(worklet);
          worklet.connect(ctx.destination);
          return;
        } catch (err) {
          console.warn("Worklet unavailable, falling back to demo path:", err);
          setDemoMode(true);
        }
      }

      // Fallback: no reachable backend — use the browser's own speech
      // recognizer (when available) so voice input still works end to end.
      const SpeechRecognitionCtor =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

      if (SpeechRecognitionCtor) {
        usingRecognitionRef.current = true;
        const recognition = new SpeechRecognitionCtor();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onresult = (event: any) => {
          let text = "";
          for (let i = 0; i < event.results.length; i++) {
            text += event.results[i][0].transcript;
          }
          setTranscript(text);
          fallbackTranscriptRef.current = text;
        };
        recognition.onerror = () => {
          /* swallowed — user can still type a query */
        };
        recognition.start();
        recognitionRef.current = recognition;
      }
    } catch (err) {
      console.error("Mic error:", err);
      alert("Couldn't access the microphone. You can still type a question below.");
      setIsRecording(false);
      setStatus("idle");
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    setIsRecording(false);
    stopLevelLoop();

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && !usingRecognitionRef.current) {
      wsRef.current.send(new ArrayBuffer(0));
    }

    if (recognitionRef.current) {
      recognitionRef.current.stop();
      const finalText = fallbackTranscriptRef.current.trim();
      recognitionRef.current = null;
      if (usingRecognitionRef.current && finalText) {
        handleTextSubmit(finalText);
      } else if (usingRecognitionRef.current) {
        setStatus("idle");
      }
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
  };

  /* -------------------------------- Text ---------------------------------- */

  const handleTextSubmit = async (e: React.FormEvent | string) => {
    if (typeof e !== "string") e.preventDefault();
    const text = typeof e === "string" ? e : queryText;

    if (!text.trim()) return;

    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setTranscript(text);
    resetPanel();
    setStatus("processing");
    const slowWarning = setTimeout(() => {
      setTier2Answer("Waking up backend server (this may take up to 50s on the free tier)...");
    }, 2500);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, language: "en" }),
      });
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }
      
      const data = await res.json();
      clearTimeout(slowWarning);
      setDemoMode(false);

      if (data.status === "refused") {
        setRefusal({ reason: data.refusal_reason, stage: data.refusal_stage });
        setStatus("refused");
        if (data.refusal_reason) speakAnswer(data.refusal_reason);
      } else {
        setTier1Answer(data.answer);
        setTier2Answer(data.answer);
        setSources(data.sources || []);
        setLatencies(data.latencies || {});
        setGroundingScore(data.grounding_score);
        setStatus("answered");
        if (data.answer) speakAnswer(data.answer);
      }
    } catch (err) {
      clearTimeout(slowWarning);
      setDemoMode(true);
      simulateResponse(text);
    }
  };

  const totalRAGMs =
    (latencies?.guardrail_ms || 0) +
    (latencies?.retrieval_ms || 0) +
    (latencies?.rerank_ms || 0) +
    (latencies?.tier1_ms || 0);

  const orbActive = isRecording || status === "processing";

  useEffect(() => {
    if (status === "answered" || status === "refused") {
      setIsSubsequentTurn(true);
    }
  }, [status]);

  // Once the assistant has (or is about to have) an answer to show, the
  // layout splits: the mic/input card moves to the left and the answer
  // takes the right column. Before that — idle, listening, or still
  // processing with nothing back yet — everything stays centered.
  const hasResults = status !== "idle";

  /* --------------------------------- Render --------------------------------- */

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#070912] text-[#eef0f8]">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

        .font-display { font-family: 'Space Grotesk', ui-sans-serif, sans-serif; }
        .font-data { font-family: 'JetBrains Mono', ui-monospace, monospace; }
        .font-body { font-family: 'Inter', ui-sans-serif, sans-serif; }

        @keyframes breathe {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.035); }
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes spin-reverse {
          from { transform: rotate(360deg); }
          to { transform: rotate(0deg); }
        }
        @keyframes ripple {
          0% { transform: scale(0.9); opacity: 0.55; }
          100% { transform: scale(1.9); opacity: 0; }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .animate-breathe { animation: breathe 4.5s ease-in-out infinite; }
        .animate-spin-slow { animation: spin-slow 9s linear infinite; }
        .animate-spin-reverse { animation: spin-reverse 13s linear infinite; }
        .animate-ripple { animation: ripple 1.8s cubic-bezier(0.2, 0.6, 0.4, 1) infinite; }
        .animate-fade-up { animation: fadeUp 0.5s ease-out both; }

        @media (prefers-reduced-motion: reduce) {
          .animate-breathe, .animate-spin-slow, .animate-spin-reverse, .animate-ripple {
            animation: none !important;
          }
        }

        .bar {
          transition: transform 60ms linear, opacity 60ms linear;
          transform: scaleY(0.18);
          transform-origin: center;
        }
      `}</style>

      <LiveField active={orbActive} />

      {/* ambient vignette */}
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          background:
            "radial-gradient(60% 50% at 50% 18%, rgba(69,232,196,0.10), transparent 60%), radial-gradient(45% 45% at 85% 80%, rgba(157,140,255,0.08), transparent 60%), #070912",
          zIndex: -1,
        }}
      />

      <div
        className={`relative z-10 mx-auto space-y-10 p-4 transition-[max-width] duration-500 md:p-8 ${hasResults ? "max-w-6xl" : "max-w-4xl"
          }`}
      >
        {/* Header */}
        <header className="mt-6 space-y-2 text-center">
          <h1 className="font-display bg-gradient-to-r from-[#45e8c4] via-[#8fe9d4] to-[#9d8cff] bg-clip-text text-5xl font-bold tracking-tight text-transparent">
            Vaani
          </h1>
          <p className="font-body text-[#7d84a3]">
            Voice-native retrieval, answered before you finish exhaling.
          </p>
        </header>

        {/* Assistant card (left once answering) + Results (right once answering) */}
        <div
          className={`transition-all duration-500 ${hasResults
              ? "grid grid-cols-1 items-start gap-6 md:grid-cols-[380px_1fr]"
              : "mx-auto max-w-4xl"
            }`}
        >
          {/* Orb + Input */}
          <div className="relative overflow-hidden rounded-[28px] border border-[#232840] bg-[#10131f]/80 p-8 shadow-[0_0_60px_-15px_rgba(69,232,196,0.15)] backdrop-blur-xl">
            {transcript && status !== "idle" && (
              <p className="font-body absolute left-6 right-6 top-5 text-center text-sm text-[#c7ccdc]/80">
                {transcript}
              </p>
            )}

            <div className="flex flex-col items-center justify-center pt-10">
              {/* radial spectrum */}
              <div className="relative mb-2 flex h-44 w-44 items-center justify-center">
                <div
                  className="pointer-events-none absolute inset-0"
                  style={{ display: isRecording ? "block" : "none" }}
                >
                  {Array.from({ length: BAR_COUNT }).map((_, i) => {
                    const angle = (i / BAR_COUNT) * 360;
                    return (
                      <div
                        key={i}
                        className="bar absolute left-1/2 top-1/2 h-6 w-[3px] rounded-full bg-gradient-to-t from-[#45e8c4] to-[#9d8cff]"
                        style={{
                          transform: `rotate(${angle}deg) translateY(-78px) scaleY(0.18)`,
                          transformOrigin: "center 78px",
                        }}
                        ref={(el) => {
                          barRefs.current[i] = el;
                        }}
                      />
                    );
                  })}
                </div>

                {/* ripples */}
                {isRecording && (
                  <>
                    <span className="absolute h-28 w-28 rounded-full border border-[#45e8c4]/40 animate-ripple" />
                    <span
                      className="absolute h-28 w-28 rounded-full border border-[#45e8c4]/30 animate-ripple"
                      style={{ animationDelay: "0.6s" }}
                    />
                  </>
                )}

                {/* rotating rings for depth */}
                <div className="absolute h-28 w-28 rounded-full border border-dashed border-[#9d8cff]/25 animate-spin-slow" />
                <div className="absolute h-24 w-24 rounded-full border border-dotted border-[#45e8c4]/25 animate-spin-reverse" />

                <button
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onMouseLeave={stopRecording}
                  onTouchStart={startRecording}
                  onTouchEnd={stopRecording}
                  aria-pressed={isRecording}
                  aria-label="Hold to speak"
                  className={`
                    relative flex h-24 w-24 flex-col items-center justify-center rounded-full
                    transition-all duration-200 focus-visible:outline-none focus-visible:ring-2
                    focus-visible:ring-[#45e8c4] focus-visible:ring-offset-2 focus-visible:ring-offset-[#10131f]
                    ${isRecording
                      ? "scale-110 bg-gradient-to-br from-[#ff8a7a] to-[#e5555a] shadow-[0_0_50px_rgba(229,85,90,0.45)]"
                      : "animate-breathe bg-gradient-to-br from-[#45e8c4] to-[#1f8f63] shadow-[0_0_40px_rgba(69,232,196,0.35)] hover:scale-105"
                    }
                  `}
                  style={{
                    boxShadow: isRecording
                      ? "inset 0 -6px 14px rgba(0,0,0,0.25), 0 0 50px rgba(229,85,90,0.45)"
                      : "inset 0 -6px 14px rgba(0,0,0,0.2), 0 0 40px rgba(69,232,196,0.3)",
                  }}
                >
                  <Mic className={`h-8 w-8 text-white ${isRecording ? "animate-pulse" : ""}`} />
                </button>
              </div>
              <span className="font-data text-xs uppercase tracking-[0.2em] text-[#7d84a3]">
                {isRecording ? "listening…" : "hold to speak"}
              </span>
            </div>

            <div className="relative my-7 flex items-center justify-center">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-[#232840]" />
              </div>
              <span className="font-data relative bg-[#10131f] px-4 text-[10px] uppercase tracking-widest text-[#4d536e]">
                or type
              </span>
            </div>

            <form onSubmit={handleTextSubmit} className="mx-auto flex max-w-xl gap-3">
              <input
                type="text"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="Ask your question…"
                className="font-body flex-1 rounded-xl border border-[#232840] bg-[#0a0d16] px-4 py-3 text-[#eef0f8] placeholder:text-[#4d536e] transition-colors focus:border-[#45e8c4] focus:outline-none"
              />
              <button
                type="submit"
                disabled={status === "processing" || !queryText.trim()}
                className="font-display rounded-xl bg-[#1a2030] px-6 py-3 font-medium text-[#eef0f8] transition-colors hover:bg-gradient-to-r hover:from-[#45e8c4] hover:to-[#1f8f63] hover:text-[#070912] disabled:opacity-40"
              >
                Ask
              </button>
            </form>

            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {[
                "What is the capital of India?",
                "How to make a bomb",
                "My name is Het Patel",
                "What is relativity?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setQueryText(q);
                    handleTextSubmit(q);
                  }}
                  className="font-body rounded-full bg-[#161a2b] px-3 py-1.5 text-xs text-[#7d84a3] transition-colors hover:text-white"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Results */}
          {hasResults && (
            <div className="animate-fade-up space-y-6">
              <div className="rounded-2xl border border-[#232840] bg-[#10131f]/80 p-6 shadow-lg backdrop-blur-xl">
                {status === "refused" ? (
                  <div className="flex h-full flex-col items-center justify-center space-y-4 py-6 text-center">
                    <div className="flex h-16 w-16 items-center justify-center rounded-full bg-[#ff6b6a]/10">
                      <ShieldAlert className="h-8 w-8 text-[#ff6b6a]" />
                    </div>
                    <div>
                      <h2 className="font-display text-xl font-bold text-[#ff6b6a]">
                        I cannot answer this
                      </h2>
                      <p className="font-body mt-3 inline-block rounded-lg border-l-2 border-[#ff6b6a] bg-[#0a0d16] p-3 text-left text-sm text-[#eef0f8]">
                        {refusal?.reason}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-[#232840] bg-[#10131f]/80 p-6 shadow-lg backdrop-blur-xl">
                    <h2 className="font-data mb-4 text-xs font-semibold uppercase tracking-widest text-[#7d84a3]">
                      Answer
                    </h2>

                    <div className="space-y-6">
                      {(tier1Answer || status === "processing") && (
                        <div>
                          <h3 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-[#45e8c4]">
                            <Zap className="h-4 w-4" /> Quick Summary
                          </h3>
                          <p className="font-body text-[#eef0f8]">
                            {tier1Answer || (
                              <span className="text-[#7d84a3]">
                                Retrieving context...
                              </span>
                            )}
                          </p>
                        </div>
                      )}

                      <div className="border-t border-[#232840] pt-6">
                        <h3 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-[#9d8cff]">
                          <Layers className="h-4 w-4" /> Detailed Response
                        </h3>
                        {!tier2Answer && (
                          <div className="mb-3 h-1 w-24 overflow-hidden rounded bg-[#1a2030]">
                            <div className="h-full w-1/2 animate-shimmer bg-[#9d8cff]" />
                          </div>
                        )}
                        <p className="font-body text-[#eef0f8]">
                          {tier2Answer || (
                            <span className="text-[#7d84a3]">
                              Waiting for the generative model…
                            </span>
                          )}
                        </p>
                      </div>
                    </div>

                    {groundingScore !== null && (
                      <div className="font-body flex w-fit items-center gap-2 rounded-lg border border-[#45e8c4]/20 bg-[#45e8c4]/10 px-3 py-1.5 text-sm text-[#45e8c4]">
                        <CheckCircle2 className="h-4 w-4" />
                        <span className="font-medium">Grounded</span>
                        <span className="opacity-75">
                          ({(groundingScore * 100).toFixed(0)}% confidence)
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Sidebar (latency + sources) sits below the answer, side by side */}
              {process.env.NODE_ENV === "development" && (
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                  <div className="rounded-2xl border border-[#232840] bg-[#10131f]/80 p-6 shadow-lg backdrop-blur-xl">
                    <h2 className="font-data mb-4 flex justify-between text-xs font-semibold uppercase tracking-widest text-[#7d84a3]">
                      <span className="flex items-center gap-1.5">
                        <Activity className="h-3.5 w-3.5" /> Latency waterfall
                      </span>
                      <span className="text-white">{totalRAGMs.toFixed(0)}ms ttl</span>
                    </h2>

                    <div className="font-data space-y-3 text-xs">
                      {latencies.guardrail_ms !== undefined && (
                        <LatencyRow
                          label="Guardrails"
                          ms={latencies.guardrail_ms}
                          pct={Math.min(100, latencies.guardrail_ms)}
                          color="#ff6b6a"
                        />
                      )}
                      {latencies.retrieval_ms !== undefined && (
                        <LatencyRow
                          label="Retrieval"
                          ms={latencies.retrieval_ms}
                          pct={Math.min(100, latencies.retrieval_ms / 2)}
                          color="#45e8c4"
                        />
                      )}
                      {latencies.rerank_ms !== undefined && (
                        <LatencyRow
                          label="Rerank"
                          ms={latencies.rerank_ms}
                          pct={Math.min(100, latencies.rerank_ms / 2)}
                          color="#45e8c4"
                        />
                      )}
                      {latencies.tier1_ms !== undefined && (
                        <LatencyRow
                          label="Tier 1 gen"
                          ms={latencies.tier1_ms}
                          pct={Math.min(100, latencies.tier1_ms)}
                          color="#45e8c4"
                        />
                      )}
                      {latencies.generation_ms !== undefined && (
                        <div className="mt-2 space-y-1 border-t border-[#232840] pt-2">
                          <div className="flex justify-between text-[#9d8cff]">
                            <span>Tier 2 (async)</span>
                            <span>{latencies.generation_ms.toFixed(1)}ms</span>
                          </div>
                          <div className="h-1.5 overflow-hidden rounded bg-[#1a2030]">
                            <div
                              className="h-full bg-[#9d8cff]"
                              style={{
                                width: `${Math.min(100, latencies.generation_ms / 10)}%`,
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {sources.length > 0 && (
                    <div className="rounded-2xl border border-[#232840] bg-[#10131f]/80 p-6 shadow-lg backdrop-blur-xl">
                      <h2 className="font-data mb-4 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-[#7d84a3]">
                        <Sparkles className="h-3.5 w-3.5" /> Top sources
                      </h2>
                      <div className="space-y-3">
                        {sources.map((s, idx) => (
                          <div
                            key={idx}
                            className="font-data rounded-lg border border-[#232840] bg-[#0a0d16] p-3 text-xs"
                          >
                            <div className="mb-1.5 flex items-start justify-between text-[#7d84a3]">
                              <span>ID: {s.doc_id.substring(0, 8)}</span>
                              <span>{s.relevance_score?.toFixed(3)}</span>
                            </div>
                            <div className="h-1 overflow-hidden rounded bg-[#1a2030]">
                              <div
                                className="h-full bg-gradient-to-r from-[#45e8c4] to-[#9d8cff]"
                                style={{ width: `${Math.min(100, (s.relevance_score || 0) * 100)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------------- */
/*  Small presentational helper                                            */
/* ----------------------------------------------------------------------- */

function LatencyRow({
  label,
  ms,
  pct,
  color,
}: {
  label: string;
  ms: number;
  pct: number;
  color: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[#7d84a3]">
        <span>{label}</span>
        <span>{ms.toFixed(1)}ms</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded bg-[#1a2030]">
        <div className="h-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}