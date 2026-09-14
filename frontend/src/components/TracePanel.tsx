"use client";

import { useState, useEffect, useRef } from "react";

interface TraceEntry {
  id: string;
  routing_path: string | null;
  classifier_label: string | null;
  classifier_confidence: number | null;
  tool_used: string | null;
  reflection_valid: boolean | null;
  reflection_reason: string | null;
  timestamp: string;
}

interface TracePanelProps {
  latestTrace: {
    tool_used: string | null;
    classifier_label: string | null;
    classifier_confidence: number | null;
    routing_path: string | null;
    reflection_valid: boolean | null;
    reflection_reason: string | null;
  } | null;
}

const bootMessages = [
  "Initializing agent runtime...",
  "8 tools registered",
  "Classifier loaded",
  "Ready",
];

export default function TracePanel({ latestTrace }: TracePanelProps) {
  const [bootComplete, setBootComplete] = useState(false);
  const [visibleBootLines, setVisibleBootLines] = useState<number[]>([]);
  const [entries, setEntries] = useState<TraceEntry[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevTraceRef = useRef<typeof latestTrace>(null);

  useEffect(() => {
    const timeouts: NodeJS.Timeout[] = [];
    bootMessages.forEach((_, index) => {
      const t = setTimeout(() => {
        setVisibleBootLines((prev) => [...prev, index]);
        if (index === bootMessages.length - 1) {
          setTimeout(() => setBootComplete(true), 400);
        }
      }, index * 500);
      timeouts.push(t);
    });
    return () => timeouts.forEach(clearTimeout);
  }, []);

  useEffect(() => {
    if (!latestTrace || latestTrace === prevTraceRef.current) return;
    prevTraceRef.current = latestTrace;

    const now = new Date();
    const timestamp = now.toTimeString().slice(0, 8);

    setEntries((prev) => [
      ...prev,
      {
        id: `${Date.now()}`,
        routing_path: latestTrace.routing_path,
        classifier_label: latestTrace.classifier_label,
        classifier_confidence: latestTrace.classifier_confidence,
        tool_used: latestTrace.tool_used,
        reflection_valid: latestTrace.reflection_valid,
        reflection_reason: latestTrace.reflection_reason,
        timestamp,
      },
    ]);
  }, [latestTrace]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  return (
    <div className="h-full flex flex-col">
      {!bootComplete ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="font-mono text-sm text-[var(--text-muted)] space-y-1">
            {bootMessages.map((msg, index) => (
              <div
                key={index}
                className={`boot-line ${visibleBootLines.includes(index) ? "opacity-100" : "opacity-0"}`}
              >
                {">"} {msg}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {entries.length === 0 && (
            <div className="font-mono text-xs text-[var(--text-muted)] pt-2">
              {">"} awaiting first request...
            </div>
          )}
          {entries.map((entry) => (
            <div
              key={entry.id}
              className="font-mono text-xs space-y-1.5 pb-3 border-b border-[var(--border-hairline)] last:border-0"
            >
              <div className="text-[var(--text-muted)]">[{entry.timestamp}]</div>

              {/* Routing */}
              <div className="flex items-center gap-2">
                <span className="text-[var(--accent-brass)]">routing:</span>
                <span>{entry.routing_path ?? "—"}</span>
              </div>

              {/* Classifier label + confidence bar */}
              {entry.classifier_label !== null && (
                <div className="flex items-center gap-2 pl-2">
                  <span className="text-[var(--text-muted)]">label:</span>
                  <span>{entry.classifier_label}</span>
                  {entry.classifier_confidence !== null && (
                    <div className="flex items-center gap-2 ml-1">
                      <div className="w-16 h-1.5 bg-[var(--bg-panel)] overflow-hidden">
                        <div
                          className="h-full bg-[var(--accent-sage)]"
                          style={{ width: `${Math.round(entry.classifier_confidence * 100)}%` }}
                        />
                      </div>
                      <span className="text-[var(--text-muted)] text-[10px]">
                        {Math.round(entry.classifier_confidence * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Tool */}
              {entry.tool_used && (
                <div className="flex items-center gap-2">
                  <span className="text-[var(--accent-brass)]">tool:</span>
                  <span>{entry.tool_used}</span>
                </div>
              )}

              {/* Reflection */}
              {entry.reflection_valid !== null && (
                <div className="flex items-start gap-2">
                  <span className="text-[var(--accent-brass)]">reflection:</span>
                  <div className="flex items-center gap-1.5">
                    <div
                      className={`w-2 h-2 rounded-full shrink-0 ${
                        entry.reflection_valid
                          ? "bg-[var(--accent-sage)]"
                          : "bg-[var(--accent-rust)]"
                      }`}
                    />
                    <span className="text-[var(--text-muted)]">
                      {entry.reflection_valid ? "valid" : "invalid"}
                    </span>
                  </div>
                  {entry.reflection_reason && (
                    <span className="text-[var(--text-muted)] text-[10px] leading-relaxed">
                      — {entry.reflection_reason}
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
