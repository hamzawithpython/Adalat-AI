"use client";

import { useState, useEffect } from "react";
import { Btn } from "@/components/ui/btn";

const STORAGE_KEY = "adalat_api_keys";

const PROVIDERS = [
  {
    id: "groq" as const,
    label: "Groq",
    placeholder: "gsk_...",
    helpUrl: "https://console.groq.com/keys",
    blurb: "Free tier - sign up at console.groq.com",
  },
  {
    id: "cerebras" as const,
    label: "Cerebras",
    placeholder: "csk-...",
    helpUrl: "https://cloud.cerebras.ai/",
    blurb: "Generous free tier - 1M tokens/day",
  },
  {
    id: "gemini" as const,
    label: "Google Gemini",
    placeholder: "AIzaSy...",
    helpUrl: "https://aistudio.google.com/app/apikey",
    blurb: "Free tier - sign up at aistudio.google.com",
  },
];

interface ApiKeysPanelProps {
  open: boolean;
  onClose: () => void;
}

export function ApiKeysPanel({ open, onClose }: ApiKeysPanelProps) {
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!open) return;
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === "object") setKeys(parsed);
      }
    } catch {}
  }, [open]);

  const handleSave = () => {
    const clean: Record<string, string> = {};
    for (const p of PROVIDERS) {
      const v = keys[p.id]?.trim();
      if (v && v.length >= 16) clean[p.id] = v;
    }
    if (Object.keys(clean).length > 0) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(clean));
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      onClose();
    }, 1200);
  };

  const handleClear = () => {
    if (!confirm("Remove all saved API keys? This will fall back to the shared rate limits.")) {
      return;
    }
    window.localStorage.removeItem(STORAGE_KEY);
    setKeys({});
  };

  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black/40 z-50" onClick={onClose} />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div className="w-full max-w-lg bg-white rounded-xl shadow-brand-lg max-h-[90vh] overflow-y-auto pointer-events-auto">
          <div className="p-6 border-b border-slate-100">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="font-serif text-2xl font-bold text-navy">Your API keys</h2>
                <p className="text-sm text-slate-600 mt-1">
                  Use your own keys to bypass shared rate limits. Keys are stored only in your browser.
                </p>
              </div>
              <button
                onClick={onClose}
                className="text-slate-400 hover:text-navy text-2xl leading-none px-2"
                aria-label="Close"
              >
                {"\u00D7"}
              </button>
            </div>
          </div>

          <div className="p-6 space-y-5">
            <div className="bg-amber-50 border border-amber-200 rounded-md px-3 py-2.5 text-[12px] text-amber-900 leading-relaxed">
              <span className="font-semibold">Note:</span>{" "}
              Keys never leave your browser except when sent to Adalat-AI&apos;s backend (which uses them only for your current request). They are not saved to our database.
            </div>

            {PROVIDERS.map((p) => (
              <div key={p.id}>
                <div className="flex items-baseline justify-between mb-1.5">
                  <label className="text-sm font-semibold text-navy">{p.label}</label>
                  <a
                    href={p.helpUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[11px] font-mono text-gold-dark hover:underline"
                  >
                    Get key {"\u2192"}
                  </a>
                </div>
                <input
                  type="password"
                  value={keys[p.id] || ""}
                  onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
                  placeholder={p.placeholder}
                  className="w-full px-3 py-2.5 text-sm font-mono rounded-md border border-slate-200 focus:border-navy focus:outline-none"
                />
                <p className="text-[11px] text-slate-500 mt-1">{p.blurb}</p>
              </div>
            ))}
          </div>

          <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between gap-3">
            <button
              onClick={handleClear}
              className="text-xs text-slate-500 hover:text-red-600"
            >
              Clear all keys
            </button>
            <div className="flex items-center gap-3">
              {saved && (
                <span className="text-sm text-green-600 font-medium">Saved</span>
              )}
              <Btn variant="primary" onClick={handleSave}>
                Save keys
              </Btn>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}