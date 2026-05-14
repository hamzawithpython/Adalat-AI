"use client";

import type { LegalResponse } from "@/types/legal";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Flag } from "@/components/brand/flag";
import { RightsCards } from "./rights-cards";
import { CitationCards } from "./citation-cards";
import { Markdown } from "@/components/ui/markdown";
import { SectionCards } from "./section-cards";
import { JudgmentCards } from "./judgment-cards";
import { FollowUpQuestions } from "./follow-up-questions";

interface AnswerViewProps {
  response: LegalResponse;
  onFollowUp?: (query: string) => void;
}

export function AnswerView({
  response,
  onFollowUp,
}: AnswerViewProps) {
  const langLabel = response.language
    .replace("_", "-")
    .toUpperCase();

  // ── Clarification branch ─────────────────────────────────
  // When the clarifier decided to ask for more facts instead of answering,
  // render a clean intake-style view. No empty Citations / Rights / Judgments
  // cards that would otherwise make the response look like a broken answer.
  if (response.is_clarification) {
    return (
      <ClarificationView
        response={response}
        langLabel={langLabel}
        onFollowUp={onFollowUp}
      />
    );
  }

  return (
    <div className="space-y-5">
      {/* Header card: meta only — confidence percentage hidden from users.
          A confident legal assistant doesn't display a self-doubting percentage. */}
      <Card padding="lg">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3 flex-wrap">
            <Flag
              code={response.jurisdiction}
              size={32}
            />

            <Badge tone="navy">
              {response.jurisdiction}
            </Badge>

            <Badge tone="goldSoft">
              {langLabel}
            </Badge>
          </div>
        </div>

        {response.translated_query &&
          response.translated_query !==
            response.query && (
            <div className="mt-4 pt-4 border-t border-slate-100">
              <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1">
                Translated query
              </div>

              <p className="text-sm text-slate-700 italic">
                “{response.translated_query}”
              </p>
            </div>
          )}
      </Card>

      {/* Main answer */}
      {response.sections &&
      response.sections.length > 0 ? (
        <SectionCards
          sections={response.sections}
        />
      ) : (
        <Card padding="lg">
          <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-3">
            Answer
          </div>

          <Markdown
            content={response.answer}
            citationCount={
              response.citations?.length ?? 0
            }
          />
        </Card>
      )}

      {/* Rights */}
      <Card padding="lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-1">
              Your Rights
            </div>

            <h3 className="font-serif text-xl font-bold text-navy">
              {response.rights.length} structured
              right
              {response.rights.length !== 1
                ? "s"
                : ""}{" "}
              extracted
            </h3>
          </div>
        </div>

        <RightsCards rights={response.rights} />
      </Card>

      {/* Citations */}
      <Card padding="lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-1">
              Citations
            </div>

            <h3 className="font-serif text-xl font-bold text-navy">
              {response.citations.length} source
              {response.citations.length !== 1
                ? "s"
                : ""}{" "}
              retrieved
            </h3>
          </div>
        </div>

        <CitationCards
          citations={response.citations}
        />
      </Card>

      {/* Judicial Principles */}
      {response.judgments &&
        response.judgments.length > 0 && (
          <Card padding="lg">
            <div className="flex items-start justify-between mb-4 gap-4 flex-wrap">
              <div>
                <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-1">
                  Judicial Principles
                </div>

                <h3 className="font-serif text-xl font-bold text-navy">
                  {
                    response.judgments.length
                  }{" "}
                  general principle
                  {response.judgments.length !== 1
                    ? "s"
                    : ""}
                </h3>
              </div>
            </div>

            {/* AI-generated disclaimer */}
            {response.judgments_disclaimer && (
              <div className="bg-amber-50 border border-amber-200 rounded-md px-3.5 py-2.5 mb-4 text-[12px] text-amber-900 leading-relaxed flex gap-2">
                <span className="font-semibold shrink-0">
                  ⚠️ Note:
                </span>

                <span>
                  {
                    response.judgments_disclaimer
                  }
                </span>
              </div>
            )}

            <JudgmentCards
              judgments={response.judgments}
            />
          </Card>
        )}

      {/* Follow-up suggestions */}
      {response.follow_up_questions &&
        response.follow_up_questions.length >
          0 &&
        onFollowUp && (
          <FollowUpQuestions
            questions={
              response.follow_up_questions
            }
            onSelect={onFollowUp}
          />
        )}

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-xs text-amber-900 leading-relaxed">
        <span className="font-semibold mr-1">
          ⚠️ Disclaimer:
        </span>

        {response.disclaimer}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// ClarificationView — shown when the assistant needs more facts
// before it can give a real legal answer.
// ─────────────────────────────────────────────────────────────
function ClarificationView({
  response,
  langLabel,
  onFollowUp,
}: {
  response: LegalResponse;
  langLabel: string;
  onFollowUp?: (q: string) => void;
}) {
  const lang = response.language;

  const title =
    lang === "roman_urdu"
      ? "Mujhe kuch baatein clear karni hain"
      : lang === "german"
        ? "Ich brauche noch ein paar Angaben"
        : "I need a few details first";

  const subtitle =
    lang === "roman_urdu"
      ? "Theek legal opinion dene ke liye, pehle yeh sawalat clear karein. Aap kisi bhi sawal par click kar sakte hain."
      : lang === "german"
        ? "Um eine fundierte rechtliche Einschätzung geben zu können, brauche ich zuerst ein paar weitere Informationen. Klicken Sie auf eine Frage, um zu antworten."
        : "Before I can give you a sound legal answer, I need to understand your situation a bit better. Click any question below to answer it.";

  const questions = response.follow_up_questions || [];

  return (
    <div className="space-y-5">
      {/* Header card: meta */}
      <Card padding="lg">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3 flex-wrap">
            <Flag code={response.jurisdiction} size={32} />
            <Badge tone="navy">{response.jurisdiction}</Badge>
            <Badge tone="goldSoft">{langLabel}</Badge>
            <Badge tone="goldSoft">CLARIFICATION</Badge>
          </div>
        </div>
      </Card>

      {/* Intake card */}
      <Card padding="lg">
        <div className="flex items-start gap-3 mb-4">
          <div className="shrink-0 flex items-center justify-center w-10 h-10 rounded-md bg-gold-faint border border-gold-soft text-gold-dark text-xl">
            ⚖️
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-1">
              Legal Intake
            </div>
            <h3 className="font-serif text-xl font-bold text-navy leading-snug">
              {title}
            </h3>
            <p className="text-sm text-slate-600 mt-1.5 leading-relaxed">
              {subtitle}
            </p>
          </div>
        </div>

        {/* Questions as a numbered list — NOT clickable.
            The user reads these and types their answers into the chat input below.
            Clicking would send the question itself as a query, which is nonsensical. */}
        {questions.length > 0 && (
          <div className="flex flex-col gap-2 mt-4">
            {questions.map((q, i) => (
              <div
                key={i}
                className="flex items-start gap-3 px-3.5 py-3 rounded-md border border-slate-200 bg-slate-50/50"
              >
                <span className="text-[10px] font-mono font-bold text-gold-dark shrink-0 mt-0.5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="text-sm text-slate-700 flex-1 leading-relaxed">
                  {q}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Hint pointing user to the input field below */}
        <div className="mt-4 pt-4 border-t border-slate-100 flex items-start gap-2 text-xs text-slate-500">
          <span className="text-base leading-none">↓</span>
          <span className="leading-relaxed">
            {lang === "roman_urdu"
              ? "Apne jawabaat neeche chat box mein likhein."
              : lang === "german"
                ? "Bitte beantworten Sie diese Fragen unten im Chatfeld."
                : "Type your answers in the chat box below."}
          </span>
        </div>
      </Card>

      {/* Subtle disclaimer */}
      <div className="text-xs text-slate-500 leading-relaxed px-1">
        {lang === "roman_urdu"
          ? "Yeh sawalat aap ko jaldi proper legal answer dene mein madad karte hain."
          : lang === "german"
            ? "Diese Fragen helfen, Ihnen schneller eine fundierte Antwort zu geben."
            : "These questions help me give you a properly grounded legal answer faster."}
      </div>
    </div>
  );
}