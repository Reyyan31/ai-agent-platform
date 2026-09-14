"use client";

import { useState } from "react";
import ConversationPanel from "@/components/ConversationPanel";
import TracePanel from "@/components/TracePanel";
import TopBar from "@/components/TopBar";

interface TraceData {
  tool_used: string | null;
  classifier_label: string | null;
  classifier_confidence: number | null;
  routing_path: string | null;
  reflection_valid: boolean | null;
  reflection_reason: string | null;
}

export default function Home() {
  const [latestTrace, setLatestTrace] = useState<TraceData | null>(null);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <TopBar />
      <main className="flex flex-1 overflow-hidden flex-col md:flex-row">
        <div
          className="flex flex-col md:w-[62%] border-b md:border-b-0 md:border-r border-[var(--border-hairline)] overflow-hidden"
          style={{ height: "60vh" }}
        >
          <ConversationPanel onTraceData={setLatestTrace} />
        </div>
        <div className="flex flex-col md:w-[38%] overflow-hidden flex-1">
          <TracePanel latestTrace={latestTrace} />
        </div>
      </main>
    </div>
  );
}
