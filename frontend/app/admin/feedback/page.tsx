"use client";

import { useState, useEffect } from "react";
import { Btn } from "@/components/ui/btn";
import { Card } from "@/components/ui/card";
import { Wordmark } from "@/components/brand/wordmark";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

interface FeedbackItem {
  id: number;
  name: string | null;
  email: string | null;
  category: string;
  message: string;
  rating: number | null;
  created_at: string;
}

export default function FeedbackAdminPage() {
  const [token, setToken] = useState("");
  const [submittedToken, setSubmittedToken] = useState("");
  const [items, setItems] = useState<FeedbackItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!submittedToken) return;
    setLoading(true);
    setError("");
    fetch(`${API_URL}/feedback/admin?token=${encodeURIComponent(submittedToken)}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setItems(data.items || []))
      .catch((e) => setError(e.message || "Failed"))
      .finally(() => setLoading(false));
  }, [submittedToken]);

  if (!submittedToken) {
    return (
      <main className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
        <Card padding="lg" className="w-full max-w-md">
          <Wordmark size="md" className="mb-6" />
          <h1 className="font-serif text-2xl font-bold text-navy mb-2">
            Admin · Feedback
          </h1>
          <p className="text-sm text-slate-600 mb-5">
            Enter your admin token to view submissions.
          </p>
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Admin token"
            className="w-full px-3 py-2.5 text-sm rounded-md border border-slate-200 focus:border-navy focus:outline-none mb-3"
            onKeyDown={(e) => e.key === "Enter" && setSubmittedToken(token)}
          />
          <Btn variant="primary" className="w-full" onClick={() => setSubmittedToken(token)}>
            View feedback
          </Btn>
        </Card>
      </main>
    );
  }

  const counts = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.category] = (acc[item.category] || 0) + 1;
    return acc;
  }, {});
  const avgRating = items.filter((i) => i.rating).length
    ? (
        items.filter((i) => i.rating).reduce((sum, i) => sum + (i.rating || 0), 0) /
        items.filter((i) => i.rating).length
      ).toFixed(1)
    : "—";

  return (
    <main className="min-h-screen bg-slate-50 p-6 lg:p-12">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-8 flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <Wordmark size="md" />
            <span className="text-slate-300">·</span>
            <span className="font-mono text-xs uppercase tracking-widest text-slate-500">
              Admin / Feedback
            </span>
          </div>
          <Btn
            variant="ghost"
            size="sm"
            onClick={() => {
              setToken("");
              setSubmittedToken("");
              setItems([]);
            }}
          >
            Sign out
          </Btn>
        </div>

        {error && (
          <Card padding="md" className="border-red-200 bg-red-50 mb-6">
            <div className="text-sm text-red-700">
              {error === "HTTP 403"
                ? "Invalid token. Try again."
                : `Error: ${error}`}
            </div>
          </Card>
        )}

        {loading && <p className="text-slate-500">Loading...</p>}

        {!loading && !error && (
          <>
            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
              <Stat label="Total" value={items.length.toString()} />
              <Stat label="Praise" value={(counts.praise || 0).toString()} emoji="🙌" />
              <Stat label="Feature" value={(counts.feature || 0).toString()} emoji="💡" />
              <Stat label="Bug" value={(counts.bug || 0).toString()} emoji="🐛" />
              <Stat label="Avg rating" value={avgRating} emoji="★" />
            </div>

            {/* List */}
            {items.length === 0 ? (
              <Card padding="lg" className="text-center text-slate-500">
                No feedback yet.
              </Card>
            ) : (
              <div className="space-y-3">
                {items.map((it) => (
                  <FeedbackRow key={it.id} item={it} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}

function Stat({ label, value, emoji }: { label: string; value: string; emoji?: string }) {
  return (
    <Card padding="md" className="text-center">
      {emoji && <div className="text-2xl mb-1">{emoji}</div>}
      <div className="text-2xl font-serif font-bold text-navy">{value}</div>
      <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mt-1">
        {label}
      </div>
    </Card>
  );
}

function FeedbackRow({ item }: { item: FeedbackItem }) {
  const date = new Date(item.created_at).toLocaleString();
  const categoryColor: Record<string, string> = {
    praise: "bg-green-100 text-green-700 border-green-200",
    feature: "bg-amber-100 text-amber-700 border-amber-200",
    bug: "bg-red-100 text-red-700 border-red-200",
    other: "bg-slate-100 text-slate-700 border-slate-200",
  };
  return (
    <Card padding="md">
      <div className="flex items-start justify-between gap-3 mb-2 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded border ${categoryColor[item.category] || categoryColor.other}`}
          >
            {item.category}
          </span>
          {item.rating && (
            <span className="text-gold text-sm">
              {"★".repeat(item.rating)}
              <span className="text-slate-200">{"★".repeat(5 - item.rating)}</span>
            </span>
          )}
          {item.name && (
            <span className="text-sm font-medium text-slate-700">{item.name}</span>
          )}
          {item.email && (
            <span className="text-xs font-mono text-slate-500">{item.email}</span>
          )}
        </div>
        <span className="text-[11px] font-mono text-slate-400">{date}</span>
      </div>
      <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
        {item.message}
      </p>
    </Card>
  );
}