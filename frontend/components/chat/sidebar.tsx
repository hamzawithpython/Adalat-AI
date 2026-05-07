"use client";

import { Wordmark } from "@/components/brand/wordmark";
import { Btn } from "@/components/ui/btn";
import { Flag } from "@/components/brand/flag";
import { JURISDICTIONS } from "@/lib/jurisdictions";
import type { Jurisdiction } from "@/types/legal";
import { cn } from "@/lib/utils";

interface SidebarProps {
  activeJurisdiction: Jurisdiction | "ALL";
  onJurisdictionChange: (j: Jurisdiction | "ALL") => void;
  onNewChat: () => void;
  isOpen?: boolean;
  onClose?: () => void;
}

export function Sidebar({
  activeJurisdiction,
  onJurisdictionChange,
  onNewChat,
  isOpen = false,
  onClose,
}: SidebarProps) {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/40 z-30"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "fixed lg:static inset-y-0 left-0 z-40 w-[280px] lg:w-[260px] h-screen bg-white border-r border-slate-200 flex flex-col transition-transform duration-200 ease-out",
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/* Wordmark + New chat */}
        <div className="p-5 border-b border-slate-100">
          <div className="flex items-center justify-between mb-5">
            <Wordmark size="md" />
            {/* Close button (mobile only) */}
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

        {/* History (placeholder) */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="text-[11px] uppercase tracking-widest font-semibold text-slate-500 mb-3 font-mono">
            Recent
          </div>
          <div className="text-sm text-slate-400 italic">
            Your past queries will appear here.
          </div>
        </div>

        {/* Footer */}
        <div className="p-5 border-t border-slate-100 text-xs text-slate-400 font-mono">
          Adalat-AI · v1.0
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
        active
          ? "bg-navy text-white font-medium"
          : "text-slate-700 hover:bg-slate-50"
      )}
    >
      {flag ?? (
        <span className="w-5 h-3.5 rounded-sm bg-gradient-to-r from-slate-300 to-slate-200" />
      )}
      <span>{label}</span>
    </button>
  );
}