"use client";

import { useState } from "react";
import { Btn } from "@/components/ui/btn";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

type Category = "bug" | "feature" | "praise" | "other";

const CATEGORIES: { value: Category; label: string; emoji: string }[] = [
  { value: "praise", label: "Praise", emoji: "🙌" },
  { value: "feature", label: "Feature idea", emoji: "💡" },
  { value: "bug", label: "Bug report", emoji: "🐛" },
  { value: "other", label: "Other", emoji: "💬" },
];

export function FeedbackForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [category, setCategory] = useState<Category>("praise");
  const [rating, setRating] = useState<number | null>(null);
  const [message, setMessage] = useState("");
  const [hover, setHover] = useState<number | null>(null);
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  const handleSubmit = async () => {
    if (!message.trim()) return;
    setStatus("submitting");
    setErrorMsg("");
    try {
      const res = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim() || null,
          email: email.trim() || null,
          category,
          message: message.trim(),
          rating,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      setStatus("success");
      // Reset form
      setName("");
      setEmail("");
      setMessage("");
      setRating(null);
      setCategory("praise");
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong");
    }
  };

  if (status === "success") {
    return (
      <Card padding="lg" className="text-center">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-green-100 text-green-700 text-2xl mb-4">
          ✓
        </div>
        <h3 className="font-serif text-2xl font-bold text-navy mb-2">
          Thank you for your feedback
        </h3>
        <p className="text-slate-600 mb-5">
          Your message has been received. We read every one.
        </p>
        <Btn variant="ghost" onClick={() => setStatus("idle")}>
          Send another
        </Btn>
      </Card>
    );
  }

  return (
    <Card padding="lg">
      <div className="space-y-5">
        {/* Category */}
        <div>
          <label className="block text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-2.5">
            What's on your mind?
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {CATEGORIES.map((c) => (
              <button
                key={c.value}
                type="button"
                onClick={() => setCategory(c.value)}
                className={cn(
                  "px-3 py-3 rounded-lg border text-sm font-medium transition-all flex flex-col items-center gap-1",
                  category === c.value
                    ? "bg-navy text-white border-navy shadow-brand-sm"
                    : "bg-white text-slate-700 border-slate-200 hover:border-slate-300"
                )}
              >
                <span className="text-lg">{c.emoji}</span>
                <span>{c.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Name + email */}
        <div className="grid sm:grid-cols-2 gap-3">
          <div>
            <label className="block text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1.5">
              Name (optional)
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={100}
              className="w-full px-3 py-2.5 text-sm rounded-md border border-slate-200 focus:border-navy focus:outline-none"
              placeholder="Your name"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1.5">
              Email (optional)
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              maxLength={200}
              className="w-full px-3 py-2.5 text-sm rounded-md border border-slate-200 focus:border-navy focus:outline-none"
              placeholder="for follow-up"
            />
          </div>
        </div>

        {/* Rating */}
        <div>
          <label className="block text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-2">
            Rate your experience (optional)
          </label>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setRating(rating === n ? null : n)}
                onMouseEnter={() => setHover(n)}
                onMouseLeave={() => setHover(null)}
                className="text-3xl transition-transform hover:scale-110"
                aria-label={`${n} stars`}
              >
                <span
                  className={cn(
                    "transition-colors",
                    (hover ?? rating ?? 0) >= n ? "text-gold" : "text-slate-200"
                  )}
                >
                  ★
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Message */}
        <div>
          <label className="block text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1.5">
            Your message
          </label>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            maxLength={5000}
            rows={5}
            placeholder="Tell us what's working, what's broken, or what you wish existed..."
            className="w-full px-3 py-2.5 text-sm rounded-md border border-slate-200 focus:border-navy focus:outline-none resize-none"
          />
          <div className="text-xs font-mono text-slate-400 text-right mt-1">
            {message.length} / 5000
          </div>
        </div>

        {/* Error */}
        {status === "error" && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-md px-3 py-2">
            {errorMsg}
          </div>
        )}

        {/* Submit */}
        <div className="flex justify-end">
          <Btn
            variant="primary"
            size="lg"
            onClick={handleSubmit}
            disabled={!message.trim() || status === "submitting"}
            iconRight={<span>→</span>}
          >
            {status === "submitting" ? "Sending..." : "Send feedback"}
          </Btn>
        </div>
      </div>
    </Card>
  );
}