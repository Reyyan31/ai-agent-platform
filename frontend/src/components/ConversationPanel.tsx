"use client";

import { useState, useRef, useEffect } from "react";
import { ImagePlus, Mic, ArrowUp, FileText, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface Message {
  id: string;
  role: "user" | "agent";
  content: string;
  loading?: boolean;
  artifact?: {
    filename: string;
    path: string;
  };
}

interface TraceData {
  tool_used: string | null;
  classifier_label: string | null;
  classifier_confidence: number | null;
  routing_path: string | null;
  reflection_valid: boolean | null;
  reflection_reason: string | null;
}

interface ConversationPanelProps {
  onTraceData: (data: TraceData) => void;
}

function isFilePath(str: string) {
  return str.endsWith(".docx") || str.endsWith(".pdf");
}

function extractFilename(path: string) {
  return path.split(/[\\/]/).pop() ?? path;
}

export default function ConversationPanel({ onTraceData }: ConversationPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pendingImage, setPendingImage] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text && !pendingImage) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
    };
    const thinkingMsg: Message = {
      id: `thinking-${Date.now()}`,
      role: "agent",
      content: "",
      loading: true,
    };

    setMessages((prev) => [...prev, userMsg, thinkingMsg]);
    setInput("");
    setPendingImage(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("message", text);
      if (pendingImage) form.append("image", pendingImage);

      const res = await fetch("/api/chat", { method: "POST", body: form });
      const data = await res.json();

      const responseText: string = data.response ?? data.error ?? "No response";
      const toolUsed: string | null = data.tool_used ?? null;

      const isArtifact =
        (toolUsed === "create_document" || toolUsed === "web_research") &&
        isFilePath(responseText);

      const agentMsg: Message = {
        id: `agent-${Date.now()}`,
        role: "agent",
        content: isArtifact ? "" : responseText,
        artifact: isArtifact
          ? { filename: extractFilename(responseText), path: responseText }
          : undefined,
      };

      setMessages((prev) => prev.filter((m) => !m.loading).concat(agentMsg));

      onTraceData({
        tool_used: toolUsed,
        classifier_label: data.classifier_label ?? null,
        classifier_confidence: data.classifier_confidence ?? null,
        routing_path: data.routing_path ?? null,
        reflection_valid: data.reflection_valid ?? null,
        reflection_reason: data.reflection_reason ?? null,
      });
    } catch {
      setMessages((prev) =>
        prev
          .filter((m) => !m.loading)
          .concat({
            id: `err-${Date.now()}`,
            role: "agent",
            content: "Something went wrong. Please try again.",
          })
      );
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setPendingImage(file);
    e.target.value = "";
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] ${
                msg.role === "user" ? "bg-[var(--bg-panel)] px-4 py-2" : "text-left"
              }`}
            >
              {msg.loading ? (
                <div className="flex items-center gap-1.5 py-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-muted)] animate-bounce [animation-delay:0ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-muted)] animate-bounce [animation-delay:150ms]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-muted)] animate-bounce [animation-delay:300ms]" />
                </div>
              ) : (
                <>
                  {msg.content && msg.role === "user" && (
                    <div className="text-sm leading-relaxed">{msg.content}</div>
                  )}
                  {msg.content && msg.role === "agent" && (
                    <div className="text-sm leading-relaxed prose-agent">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code({ className, children, ...props }) {
                            const match = /language-(\w+)/.exec(className || "");
                            const isBlock = !!match;
                            return isBlock ? (
                              <SyntaxHighlighter
                                style={oneDark}
                                language={match[1]}
                                PreTag="div"
                                customStyle={{
                                  margin: "0.5rem 0",
                                  fontSize: "0.75rem",
                                  background: "var(--bg-panel)",
                                  border: "1px solid var(--border-hairline)",
                                  borderRadius: 0,
                                }}
                              >
                                {String(children).replace(/\n$/, "")}
                              </SyntaxHighlighter>
                            ) : (
                              <code
                                className="px-1 py-0.5 text-[0.8em]"
                                style={{
                                  background: "var(--bg-panel)",
                                  fontFamily: "var(--font-mono)",
                                  color: "var(--accent-brass)",
                                }}
                                {...props}
                              >
                                {children}
                              </code>
                            );
                          },
                          h1: ({ children }) => (
                            <h1 className="text-base font-semibold mt-3 mb-1 text-[var(--text-primary)]">{children}</h1>
                          ),
                          h2: ({ children }) => (
                            <h2 className="text-sm font-semibold mt-3 mb-1 text-[var(--text-primary)]">{children}</h2>
                          ),
                          h3: ({ children }) => (
                            <h3 className="text-sm font-medium mt-2 mb-1 text-[var(--text-primary)]">{children}</h3>
                          ),
                          p: ({ children }) => (
                            <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
                          ),
                          ul: ({ children }) => (
                            <ul className="mb-2 space-y-0.5 pl-4 list-disc marker:text-[var(--text-muted)]">{children}</ul>
                          ),
                          ol: ({ children }) => (
                            <ol className="mb-2 space-y-0.5 pl-4 list-decimal marker:text-[var(--text-muted)]">{children}</ol>
                          ),
                          li: ({ children }) => (
                            <li className="leading-relaxed">{children}</li>
                          ),
                          strong: ({ children }) => (
                            <strong className="font-semibold text-[var(--text-primary)]">{children}</strong>
                          ),
                          blockquote: ({ children }) => (
                            <blockquote className="border-l-2 border-[var(--border-hairline)] pl-3 text-[var(--text-muted)] my-2">{children}</blockquote>
                          ),
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                  {msg.artifact && (
                    <div className="mt-2 border border-[var(--border-hairline)] p-3 flex items-center gap-3">
                      <FileText className="w-5 h-5 text-[var(--accent-brass)] shrink-0" />
                      <span className="text-sm text-[var(--text-primary)] truncate">
                        {msg.artifact.filename}
                      </span>
                      <a
                        href={`/api/download?filename=${encodeURIComponent(msg.artifact.filename)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-[var(--accent-brass)] ml-auto hover:underline"
                      >
                        Download
                      </a>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Composer */}
      <div className="border-t border-[var(--border-hairline)]">
        {pendingImage && (
          <div className="px-3 pt-2 flex items-center gap-2">
            <div className="flex items-center gap-2 bg-[var(--bg-panel)] px-2 py-1 text-xs text-[var(--text-muted)]">
              <FileText className="w-3.5 h-3.5" />
              <span className="max-w-[160px] truncate">{pendingImage.name}</span>
              <button
                onClick={() => setPendingImage(null)}
                className="ml-1 hover:text-[var(--text-primary)]"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
        <div className="p-3 flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            onClick={handleImageUpload}
            className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <ImagePlus className="w-5 h-5" />
          </button>
          <button
            onClick={() => console.log("Mic clicked")}
            className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <Mic className="w-5 h-5" />
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !loading && handleSend()}
            placeholder="Ask anything..."
            disabled={loading}
            className="flex-1 bg-transparent border-none outline-none text-[var(--text-primary)] placeholder:text-[var(--text-muted)] disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={loading || (!input.trim() && !pendingImage)}
            className="p-2 text-[var(--accent-brass)] hover:text-[var(--text-primary)] transition-colors disabled:opacity-40"
          >
            <ArrowUp className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
