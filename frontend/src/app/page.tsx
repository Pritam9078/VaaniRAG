"use client";

import { useState, useRef } from "react";
import { Mic, Loader2, CheckCircle2, AlertTriangle } from "lucide-react";
import { QueryClient, QueryClientProvider, useMutation } from "@tanstack/react-query";

const queryClient = new QueryClient();

interface Latencies {
  stt_ms: number;
  retrieval_ms: number;
  rerank_ms: number;
  generation_ms: number;
  total_ms: number;
}

interface Source {
  chunk_id: string;
  doc_id: string;
}

interface AskResponse {
  status: "answered" | "refused";
  answer?: string;
  refusal_reason?: string;
  grounding_score?: number;
  sources: Source[];
  latencies: Latencies;
}

function MainInterface() {
  const [isRecording, setIsRecording] = useState(false);
  const [queryText, setQueryText] = useState("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const askMutation = useMutation({
    mutationFn: async (payload: { audio?: Blob; text?: string }) => {
      let url = "";
      let options: RequestInit = { method: "POST" };

      if (payload.audio) {
        url = "http://localhost:8000/ask";
        const formData = new FormData();
        formData.append("audio", payload.audio, "recording.webm");
        options.body = formData;
      } else if (payload.text) {
        url = "http://localhost:8000/query";
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify({ query: payload.text, language: "en" });
      }

      // Fallback for now to point to current running endpoints if api/v1/ask doesn't exist
      url = payload.audio ? "http://localhost:8000/ask" : "http://localhost:8000/query";

      const response = await fetch(url, options);
      if (!response.ok) throw new Error("API failed");
      return response.json() as Promise<AskResponse>;
    },
  });

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        askMutation.mutate({ audio: audioBlob });
        stream.getTracks().forEach((track) => track.stop());
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error("Mic error:", err);
      alert("Microphone access denied or unavailable.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (queryText.trim()) {
      askMutation.mutate({ text: queryText });
    }
  };

  const data = askMutation.data;
  const isLoading = askMutation.isPending;

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e9edf1] font-sans p-4 md:p-8">
      <div className="max-w-3xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-2 mt-8">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-[#3ddc97] to-[#1f8f63] text-transparent bg-clip-text">
            VaaniRAG
          </h1>
          <p className="text-[#8b98a5] text-lg">Voice-powered knowledge assistant</p>
        </div>

        {/* Input Area */}
        <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-8 text-center space-y-6 shadow-xl">
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={stopRecording}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
            className={`
              w-32 h-32 rounded-full mx-auto flex flex-col items-center justify-center transition-all duration-200
              ${isRecording 
                ? "bg-[#e5555a] scale-110 shadow-[0_0_40px_rgba(229,85,90,0.4)]" 
                : "bg-gradient-to-br from-[#3ddc97] to-[#1f8f63] hover:scale-105 shadow-lg"}
            `}
          >
            <Mic className={`w-12 h-12 ${isRecording ? "text-white animate-pulse" : "text-white"}`} />
            <span className="text-white/90 text-sm mt-2 font-medium tracking-wide">
              {isRecording ? "Listening..." : "Hold Speak"}
            </span>
          </button>

          <div className="relative flex items-center justify-center">
            <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-[#232c37]" /></div>
            <span className="relative bg-[#141b24] px-4 text-xs uppercase tracking-widest text-[#8b98a5]">OR</span>
          </div>

          <form onSubmit={handleTextSubmit} className="flex gap-3">
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="Ask your question..."
              className="flex-1 bg-[#0d1117] border border-[#232c37] rounded-xl px-4 py-3 text-white focus:outline-none focus:border-[#3ddc97] transition-colors"
            />
            <button
              type="submit"
              disabled={isLoading || !queryText.trim()}
              className="bg-[#232c37] hover:bg-[#3ddc97] hover:text-white text-[#e9edf1] px-6 py-3 rounded-xl font-medium transition-colors disabled:opacity-50"
            >
              Ask
            </button>
          </form>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-12 text-[#8b98a5]">
            <Loader2 className="w-8 h-8 animate-spin" />
            <span className="ml-3 font-medium">Processing query...</span>
          </div>
        )}

        {/* Results Area */}
        {data && !isLoading && (
          <div className="space-y-6 animate-in fade-in duration-500">
            
            {/* Answer Box */}
            <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-6 shadow-lg">
              <h2 className="text-xs uppercase tracking-widest text-[#8b98a5] mb-4 font-semibold">Answer</h2>
              
              {data.status === "answered" ? (
                <div className="space-y-6">
                  <p className="text-lg leading-relaxed text-[#e9edf1]">
                    {data.answer}
                  </p>
                  {data.grounding_score !== undefined && (
                    <div className="flex items-center gap-2 text-sm text-[#3ddc97] bg-[#3ddc97]/10 w-fit px-3 py-1.5 rounded-lg border border-[#3ddc97]/20">
                      <CheckCircle2 className="w-4 h-4" />
                      <span className="font-medium">Grounded</span>
                      <span className="opacity-75">({(data.grounding_score * 100).toFixed(0)}% confidence)</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-start gap-3 text-[#e5555a] bg-[#e5555a]/10 p-4 rounded-xl border border-[#e5555a]/20">
                  <AlertTriangle className="w-5 h-5 mt-0.5 flex-none" />
                  <div>
                    <p className="font-medium">Refused to answer</p>
                    <p className="text-sm opacity-90 mt-1">{data.refusal_reason}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Sources */}
            {data.sources && data.sources.length > 0 && (
              <div className="bg-[#141b24] rounded-2xl border border-[#232c37] p-6 shadow-lg">
                <h2 className="text-xs uppercase tracking-widest text-[#8b98a5] mb-4 font-semibold">Sources</h2>
                <div className="flex flex-wrap gap-2">
                  {data.sources.map((s, idx) => (
                    <div key={idx} className="bg-[#0d1117] border border-[#232c37] text-sm px-3 py-1.5 rounded-lg text-[#8b98a5]">
                      Document {s.doc_id.replace("doc_", "")}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Performance Dashboard */}
            <div className="bg-[#0d1117] rounded-2xl border border-[#232c37] p-6 shadow-lg">
              <h2 className="text-xs uppercase tracking-widest text-[#8b98a5] mb-6 font-semibold flex items-center justify-between">
                Performance
                <span className="bg-[#3ddc97]/10 text-[#3ddc97] px-2 py-0.5 rounded text-[10px]">LIVE</span>
              </h2>
              
              <div className="space-y-3 font-mono text-sm mb-6">
                {data.latencies.stt_ms > 0 && (
                  <div className="flex justify-between items-center text-[#8b98a5]">
                    <span>Speech-to-Text</span>
                    <span className="text-[#e9edf1]">{data.latencies.stt_ms.toFixed(0)} ms</span>
                  </div>
                )}
                <div className="flex justify-between items-center text-[#8b98a5]">
                  <span>Retrieval & Rerank</span>
                  <span className="text-[#e9edf1]">{(data.latencies.retrieval_ms + data.latencies.rerank_ms).toFixed(1)} ms</span>
                </div>
                <div className="flex justify-between items-center text-[#8b98a5]">
                  <span>Generation</span>
                  <span className="text-[#e9edf1]">{data.latencies.generation_ms.toFixed(1)} ms</span>
                </div>
                
                <div className="pt-3 mt-3 border-t border-[#232c37] flex justify-between items-center font-bold">
                  <span className="text-white">RAG Total</span>
                  <span className="text-[#3ddc97]">{data.latencies.total_ms.toFixed(1)} ms</span>
                </div>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MainInterface />
    </QueryClientProvider>
  );
}
