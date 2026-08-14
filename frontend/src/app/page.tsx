"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Loader2, CheckCircle2, AlertTriangle, Zap, Server } from "lucide-react";

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

export default function MainInterface() {
  const [isRecording, setIsRecording] = useState(false);
  const [queryText, setQueryText] = useState("");
  
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState<"idle" | "listening" | "processing" | "answered" | "refused">("idle");
  const [tier1Answer, setTier1Answer] = useState("");
  const [tier2Answer, setTier2Answer] = useState("");
  const [refusal, setRefusal] = useState<{reason: string, stage: string} | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [latencies, setLatencies] = useState<Latencies>({});
  const [groundingScore, setGroundingScore] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  // Load Worklet on mount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  const connectWebSocket = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        resolve(wsRef.current);
        return;
      }
      
      const ws = new WebSocket("ws://localhost:8000/ws/voice");
      ws.binaryType = "arraybuffer";
      
      ws.onopen = () => resolve(ws);
      ws.onerror = (e) => reject(e);
      
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

  const handleWebSocketMessage = (msg: WSMessage) => {
    switch (msg.type) {
      case "transcript_partial":
        setTranscript(msg.text || "");
        if (msg.is_final) setStatus("processing");
        break;
      case "tier1":
        setTier1Answer(msg.answer || "");
        if (msg.sources) setSources(msg.sources);
        if (msg.latencies) setLatencies(prev => ({ ...prev, ...msg.latencies }));
        setStatus("answered");
        break;
      case "tier2":
        setTier2Answer(msg.answer || "");
        if (msg.grounding_score) setGroundingScore(msg.grounding_score);
        if (msg.latencies) setLatencies(prev => ({ ...prev, ...msg.latencies }));
        break;
      case "refused":
      case "tier2_refused":
        setRefusal({ reason: msg.reason || "Unknown", stage: msg.stage || "generation" });
        setStatus("refused");
        break;
      case "error":
      case "tier2_error":
        console.error("Server Error:", msg.message);
        setRefusal({ reason: msg.message || "Server Error", stage: "error" });
        setStatus("refused");
        break;
    }
  };

  const startRecording = async () => {
    try {
      setTranscript("");
      setTier1Answer("");
      setTier2Answer("");
      setRefusal(null);
      setSources([]);
      setLatencies({});
      setGroundingScore(null);
      
      const ws = await connectWebSocket();
      setStatus("listening");
      setIsRecording(true);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000 }
      });
      streamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = ctx;

      await ctx.audioWorklet.addModule("/audio-worklet.js");
      
      const source = ctx.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(ctx, "pcm-capture");
      workletNodeRef.current = worklet;

      worklet.port.onmessage = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(e.data); // e.data is ArrayBuffer (PCM16)
        }
      };

      source.connect(worklet);
      worklet.connect(ctx.destination);
    } catch (err) {
      console.error("Mic/WS error:", err);
      alert("Error accessing microphone or connecting to server.");
      setIsRecording(false);
      setStatus("idle");
    }
  };

  const stopRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Send EOF to flush STT
        wsRef.current.send(new ArrayBuffer(0));
      }

      if (workletNodeRef.current) {
        workletNodeRef.current.disconnect();
        workletNodeRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    }
  };

  const handleTextSubmit = async (e: React.FormEvent | string) => {
    if (typeof e !== "string") e.preventDefault();
    const text = typeof e === "string" ? e : queryText;
    
    if (!text.trim()) return;
    
    setTranscript(text);
    setTier1Answer("");
    setTier2Answer("");
    setRefusal(null);
    setSources([]);
    setLatencies({});
    setGroundingScore(null);
    setStatus("processing");
    
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, language: "en" })
      });
      const data = await res.json();
      
      if (data.status === "refused") {
        setRefusal({ reason: data.refusal_reason, stage: data.refusal_stage });
        setStatus("refused");
      } else {
        setTier2Answer(data.answer);
        setSources(data.sources);
        setLatencies(data.latencies);
        setGroundingScore(data.grounding_score);
        setStatus("answered");
      }
    } catch (err) {
      console.error(err);
      setStatus("refused");
      setRefusal({ reason: "HTTP Error", stage: "network" });
    }
  };

  const totalRAGMs = (latencies.guardrail_ms || 0) + (latencies.retrieval_ms || 0) + (latencies.rerank_ms || 0) + (latencies.tier1_ms || 0);

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e9edf1] font-sans p-4 md:p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-2 mt-4">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-[#3ddc97] to-[#1f8f63] text-transparent bg-clip-text">
            VaaniRAG
          </h1>
          <p className="text-[#8b98a5] text-lg">Ultra-Low Latency Voice Assistant</p>
        </div>

        {/* Input Area */}
        <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-8 text-center space-y-6 shadow-xl relative overflow-hidden">
          
          {/* Transcript overlay */}
          {transcript && status !== "idle" && (
            <div className="absolute top-4 left-4 right-4 text-center">
              <p className="text-xl font-medium text-white/80">{transcript}</p>
            </div>
          )}

          <div className="pt-8">
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              className={`
                w-28 h-28 rounded-full mx-auto flex flex-col items-center justify-center transition-all duration-200
                ${isRecording 
                  ? "bg-[#e5555a] scale-110 shadow-[0_0_40px_rgba(229,85,90,0.4)]" 
                  : "bg-gradient-to-br from-[#3ddc97] to-[#1f8f63] hover:scale-105 shadow-lg"}
              `}
            >
              <Mic className={`w-10 h-10 ${isRecording ? "text-white animate-pulse" : "text-white"}`} />
              <span className="text-white/90 text-xs mt-2 font-medium tracking-wide">
                {isRecording ? "Listening..." : "Hold Speak"}
              </span>
            </button>
          </div>

          <div className="relative flex items-center justify-center">
            <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-[#232c37]" /></div>
            <span className="relative bg-[#141b24] px-4 text-xs uppercase tracking-widest text-[#8b98a5]">OR</span>
          </div>

          <form onSubmit={handleTextSubmit} className="flex gap-3 max-w-xl mx-auto">
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="Ask your question..."
              className="flex-1 bg-[#0d1117] border border-[#232c37] rounded-xl px-4 py-3 text-white focus:outline-none focus:border-[#3ddc97] transition-colors"
            />
            <button
              type="submit"
              disabled={status === "processing" || !queryText.trim()}
              className="bg-[#232c37] hover:bg-[#3ddc97] hover:text-white text-[#e9edf1] px-6 py-3 rounded-xl font-medium transition-colors disabled:opacity-50"
            >
              Ask
            </button>
          </form>

          {/* Sample Chips */}
          <div className="flex flex-wrap justify-center gap-2 pt-2">
            {[
              "What is the capital of India?",
              "How to make a bomb", // Triggers safety
              "My name is Het Patel", // Triggers scope
              "What is relativity?" // Might trigger weak retrieval depending on index
            ].map(q => (
              <button 
                key={q} 
                onClick={() => { setQueryText(q); handleTextSubmit(q); }}
                className="text-xs bg-[#232c37] text-[#8b98a5] hover:text-white px-3 py-1.5 rounded-full transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Results Area */}
        {(status === "answered" || status === "refused" || tier1Answer) && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-in fade-in duration-500">
            
            {/* Main Content Area (Answer/Refusal) */}
            <div className="md:col-span-2 space-y-6">
              <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-6 shadow-lg h-full">
                
                {status === "refused" ? (
                  <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                     {/* Refusal Lamp */}
                     <div className="w-16 h-16 rounded-full bg-[#e5555a]/20 flex items-center justify-center">
                        <div className="w-8 h-8 rounded-full bg-[#e5555a] shadow-[0_0_30px_rgba(229,85,90,0.8)] animate-pulse" />
                     </div>
                     <div>
                        <h2 className="text-xl font-bold text-[#e5555a]">Guardrail Refusal</h2>
                        <p className="text-[#8b98a5] mt-2">Stage: <span className="font-mono text-white">{refusal?.stage}</span></p>
                        <p className="text-white mt-2 border-l-2 border-[#e5555a] pl-3 text-left inline-block bg-[#0d1117] p-3 rounded">{refusal?.reason}</p>
                     </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <h2 className="text-xs uppercase tracking-widest text-[#8b98a5] font-semibold">Two-Tier Generation</h2>
                    
                    {/* Tier 1 Box */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Zap className="w-4 h-4 text-[#3ddc97]" />
                        <span className="text-sm font-medium text-[#3ddc97]">Tier 1: Extractive (Fast)</span>
                      </div>
                      <p className="text-lg leading-relaxed text-[#8b98a5] italic border-l-2 border-[#232c37] pl-3">
                        {tier1Answer}
                      </p>
                    </div>

                    {/* Tier 2 Box */}
                    <div className="space-y-2 pt-4 border-t border-[#232c37]">
                      <div className="flex items-center gap-2">
                        <Server className="w-4 h-4 text-[#a371f7]" />
                        <span className="text-sm font-medium text-[#a371f7]">Tier 2: Generative (Reasoned)</span>
                        {!tier2Answer && <Loader2 className="w-3 h-3 animate-spin text-[#a371f7]" />}
                      </div>
                      <p className="text-lg leading-relaxed text-white min-h-[3rem]">
                        {tier2Answer || <span className="text-[#8b98a5]/50 italic">Waiting for LLM generation...</span>}
                      </p>
                    </div>

                    {groundingScore !== null && (
                      <div className="flex items-center gap-2 text-sm text-[#3ddc97] bg-[#3ddc97]/10 w-fit px-3 py-1.5 rounded-lg border border-[#3ddc97]/20">
                        <CheckCircle2 className="w-4 h-4" />
                        <span className="font-medium">Grounded</span>
                        <span className="opacity-75">({(groundingScore * 100).toFixed(0)}% confidence)</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Sidebar (Waterfall + Sources) */}
            <div className="space-y-6">
              
              {/* Latency Waterfall */}
              <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-6 shadow-lg">
                <h2 className="text-xs uppercase tracking-widest text-[#8b98a5] mb-4 font-semibold flex justify-between">
                  Latency Waterfall
                  <span className="text-white font-mono">{totalRAGMs.toFixed(0)}ms TTL</span>
                </h2>
                
                <div className="space-y-3 font-mono text-xs">
                  {latencies.guardrail_ms !== undefined && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[#8b98a5]"><span>Guardrails</span> <span>{latencies.guardrail_ms.toFixed(1)}ms</span></div>
                      <div className="h-1.5 bg-[#232c37] rounded overflow-hidden">
                        <div className="h-full bg-[#e5555a]" style={{ width: `${Math.min(100, latencies.guardrail_ms)}%` }} />
                      </div>
                    </div>
                  )}
                  {latencies.retrieval_ms !== undefined && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[#8b98a5]"><span>Retrieval</span> <span>{latencies.retrieval_ms.toFixed(1)}ms</span></div>
                      <div className="h-1.5 bg-[#232c37] rounded overflow-hidden">
                        <div className="h-full bg-[#3ddc97]" style={{ width: `${Math.min(100, latencies.retrieval_ms / 2)}%` }} />
                      </div>
                    </div>
                  )}
                  {latencies.rerank_ms !== undefined && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[#8b98a5]"><span>Rerank</span> <span>{latencies.rerank_ms.toFixed(1)}ms</span></div>
                      <div className="h-1.5 bg-[#232c37] rounded overflow-hidden">
                        <div className="h-full bg-[#3ddc97]" style={{ width: `${Math.min(100, latencies.rerank_ms / 2)}%` }} />
                      </div>
                    </div>
                  )}
                  {latencies.tier1_ms !== undefined && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[#8b98a5]"><span>Tier 1 Gen</span> <span>{latencies.tier1_ms.toFixed(1)}ms</span></div>
                      <div className="h-1.5 bg-[#232c37] rounded overflow-hidden">
                        <div className="h-full bg-[#3ddc97]" style={{ width: `${Math.min(100, latencies.tier1_ms)}%` }} />
                      </div>
                    </div>
                  )}
                  {latencies.generation_ms !== undefined && (
                    <div className="space-y-1 pt-2 mt-2 border-t border-[#232c37]">
                      <div className="flex justify-between text-[#a371f7]"><span>Tier 2 (Async)</span> <span>{latencies.generation_ms.toFixed(1)}ms</span></div>
                      <div className="h-1.5 bg-[#232c37] rounded overflow-hidden">
                        <div className="h-full bg-[#a371f7]" style={{ width: `${Math.min(100, latencies.generation_ms / 10)}%` }} />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Sources */}
              {sources.length > 0 && (
                <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-6 shadow-lg">
                  <h2 className="text-xs uppercase tracking-widest text-[#8b98a5] mb-4 font-semibold">Top Sources</h2>
                  <div className="space-y-3">
                    {sources.map((s, idx) => (
                      <div key={idx} className="bg-[#0d1117] border border-[#232c37] p-3 rounded-lg text-sm">
                        <div className="flex justify-between items-start text-[#8b98a5] mb-1">
                          <span className="font-mono text-xs">ID: {s.doc_id.substring(0,8)}</span>
                          <span className="text-xs">Score: {s.relevance_score?.toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          </div>
        )}
      </div>
    </div>
  );
}
