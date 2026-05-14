"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Wordmark } from "@/components/brand/wordmark";
import { Btn } from "@/components/ui/btn";
import { Flag } from "@/components/brand/flag";
import { JURISDICTIONS } from "@/lib/jurisdictions";
import type { Jurisdiction } from "@/types/legal";
import type { SessionListItem } from "@/lib/api";
import { cn } from "@/lib/utils";

interface SidebarProps {
  activeJurisdiction: Jurisdiction | "ALL";
  onJurisdictionChange: (j: Jurisdiction | "ALL") => void;
  onNewChat: () => void;
  isOpen?: boolean;
  onClose?: () => void;
  // History
  sessions: SessionListItem[];
  historyLoading: boolean;
  historyError: string | null;
  activeSessionId: string | null;
  onSessionSelect: (id: string) => void;
  onSessionDelete: (id: string) => void;
  onHistoryRefresh: () => void;
  onClearAll: () => void;
  clearing?: boolean;
  onOpenSettings?: () => void;
}

export function Sidebar({
  activeJurisdiction,
  onJurisdictionChange,
  onNewChat,
  isOpen = false,
  onClose,
  sessions,
  historyLoading,
  historyError,
  activeSessionId,
  onSessionSelect,
  onSessionDelete,
  onHistoryRefresh,
  onClearAll,
  clearing = false,
  onOpenSettings,
}: SidebarProps) {

  return (
    <>
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/40 z-30"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "fixed lg:static inset-y-0 left-0 z-40 w-[280px] lg:w-[280px] h-screen bg-white border-r border-slate-200 flex flex-col transition-transform duration-200 ease-out",
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/* Wordmark + New chat */}
        <div className="p-5 border-b border-slate-100">
          <div className="flex items-center justify-between mb-5">
            <Link
              href="/"
              className="inline-block hover:opacity-80 transition-opacity"
              aria-label="Go to home page"
            >
              <Wordmark size="md" />
            </Link>
            <button
              onClick={onClose}
              className="lg:hidden text-slate-400 hover:text-navy text-xl leading-none px-2"
              aria-label="Close sidebar"
            >
              ×
            </button>
          </div>
          <Btn
            variant="primary"
            size="sm"
            className="w-full"
            iconRight={<span>+</span>}
            onClick={() => {
              onNewChat();
              onClose?.();
            }}
          >
            New chat
          </Btn>
        </div>

        {/* Jurisdiction filter */}
        <div className="p-5 border-b border-slate-100">
          <div className="text-[11px] uppercase tracking-widest font-semibold text-slate-500 mb-3 font-mono">
            Jurisdiction
          </div>
          <div className="flex flex-col gap-1">
            <JurisdictionPill
              label="All jurisdictions"
              active={activeJurisdiction === "ALL"}
              onClick={() => onJurisdictionChange("ALL")}
            />
            {JURISDICTIONS.map((j) => (
              <JurisdictionPill
                key={j.code}
                label={j.name}
                flag={<Flag code={j.code} size={20} />}
                active={activeJurisdiction === j.code}
                onClick={() => onJurisdictionChange(j.code)}
              />
            ))}
          </div>
        </div>

        {/* History */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="text-[11px] uppercase tracking-widest font-semibold text-slate-500 font-mono">
              Recent
            </div>
            <div className="flex items-center gap-1">
              {sessions.length > 0 && (
                <ClearAllButton
                  onConfirm={onClearAll}
                  disabled={clearing}
                  count={sessions.length}
                />
              )}
              <button
                onClick={onHistoryRefresh}
                className="text-slate-400 hover:text-navy text-sm px-1.5"
                aria-label="Refresh"
                title="Refresh"
              >
                ↻
              </button>
            </div>
          </div>

          {historyLoading && sessions.length === 0 && (
            <div className="text-sm text-slate-400 italic">Loading…</div>
          )}
          {historyError && (
            <div className="text-xs text-red-600">{historyError}</div>
          )}
          {!historyLoading && !historyError && sessions.length === 0 && (
            <div className="text-sm text-slate-400 italic">
              Your past chats will appear here.
            </div>
          )}

          <div className="flex flex-col gap-1">
            {sessions.map((s) => (
              <SessionRow
                key={s.id}
                session={s}
                active={activeSessionId === s.id}
                onClick={() => {
                  onSessionSelect(s.id);
                  onClose?.();
                }}
                onDelete={() => onSessionDelete(s.id)}
              />
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-5 border-t border-slate-100 flex items-center justify-between">
          <button
            onClick={onOpenSettings}
            className="text-xs text-slate-500 hover:text-navy font-medium flex items-center gap-1.5"
          >
            <span>🔑</span>
            <span>API keys</span>
          </button>
          <span className="text-xs text-slate-400 font-mono">v1.0</span>
        </div>
      </aside>
    </>
  );
}

function JurisdictionPill({
  label,
  flag,
  active,
  onClick,
}: {
  label: string;
  flag?: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2.5 px-3 py-2 rounded-md text-sm text-left transition-colors",
        active ? "bg-navy text-white font-medium" : "text-slate-700 hover:bg-slate-50"
      )}
    >
      {flag ?? (
        <span className="w-5 h-3.5 rounded-sm bg-gradient-to-r from-slate-300 to-slate-200" />
      )}
      <span>{label}</span>
    </button>
  );
}

function SessionRow({
  session,
  active,
  onClick,
  onDelete,
}: {
  session: SessionListItem;
  active: boolean;
  onClick: () => void;
  onDelete: () => void;
}) {
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm("Delete this chat?")) onDelete();
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        "group relative px-3 py-2 rounded-md cursor-pointer transition-colors",
        active ? "bg-gold-faint" : "hover:bg-slate-50"
      )}
    >
      <div className="flex items-start gap-2">
        {session.jurisdiction && (
          <Flag code={session.jurisdiction} size={16} />
        )}
        <div className="flex-1 min-w-0">
          <div
            className={cn(
              "text-[13px] leading-snug truncate",
              active ? "text-navy font-medium" : "text-slate-700"
            )}
          >
            {session.title || "Untitled chat"}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5 text-[10px] font-mono text-slate-400">
            <span>{session.turn_count} turn{session.turn_count !== 1 ? "s" : ""}</span>
            <span>·</span>
            <span>{formatDate(session.updated_at)}</span>
          </div>
        </div>
      </div>
      <button
        onClick={handleDelete}
        className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 text-xs px-1.5 transition-opacity"
        aria-label="Delete"
        title="Delete"
      >
        ×
      </button>
    </div>
  );
}

function ClearAllButton({
  onConfirm,
  disabled,
  count,
}: {
  onConfirm: () => void;
  disabled: boolean;
  count: number;
}) {
  const [armed, setArmed] = useState(false);

  // Auto-disarm after 4 seconds
  useEffect(() => {
    if (!armed) return;
    const id = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(id);
  }, [armed]);

  if (disabled) {
    return (
      <span className="text-[10px] font-mono text-slate-400 px-2">
        Clearing…
      </span>
    );
  }

  if (armed) {
    return (
      <button
        onClick={() => {
          onConfirm();
          setArmed(false);
        }}
        className="text-[10px] font-mono uppercase tracking-wider text-red-600 hover:text-red-700 font-semibold px-2 py-1 rounded border border-red-200 bg-red-50 hover:bg-red-100 transition-colors"
        title={`Confirm: delete all ${count} chats`}
      >
        Confirm
      </button>
    );
  }

  return (
    <button
      onClick={() => setArmed(true)}
      className="text-[10px] font-mono uppercase tracking-wider text-slate-400 hover:text-red-600 px-1.5 py-1 transition-colors"
      title={`Clear all ${count} recent chats`}
    >
      Clear all
    </button>
  );
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMs / 3600000);
  const diffDay = Math.floor(diffMs / 86400000);

  if (diffMin < 1) return "now";
  if (diffMin < 60) return `${diffMin}m`;
  if (diffHr < 24) return `${diffHr}h`;
  if (diffDay < 7) return `${diffDay}d`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}