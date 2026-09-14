export default function TopBar() {
  return (
    <header className="h-12 border-b border-[var(--border-hairline)] flex items-center justify-between px-4">
      <div className="flex items-center gap-3">
        <span className="text-lg font-semibold tracking-tight text-[var(--text-primary)]">
          Cognitive Agent
        </span>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--accent-sage)]" />
          <span className="text-xs text-[var(--text-muted)]">connected</span>
        </div>
        <div className="text-xs px-2 py-1 bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[var(--text-muted)]">
          model: groq
        </div>
      </div>
    </header>
  );
}