"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

interface MarkdownProps {
  content: string;
  className?: string;
  /**
   * Optional citation count. When provided, [^N] markers in the content
   * (where 1 ≤ N ≤ citationCount) are rewritten as clickable superscript
   * links that scroll to the matching citation card (#citation-N).
   */
  citationCount?: number;
}

function CitationAnchor({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      onClick={(e) => {
        e.preventDefault();
        const target = document.getElementById(href.slice(1));
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.classList.add("ring-2", "ring-navy", "ring-offset-2");
          setTimeout(() => {
            target.classList.remove("ring-2", "ring-navy", "ring-offset-2");
          }, 1500);
        }
      }}
      className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 ml-0.5 text-[10px] font-mono font-bold text-white bg-navy rounded hover:bg-gold-dark transition-colors no-underline align-super"
    >
      {children}
    </a>
  );
}

export function Markdown({ content, className, citationCount }: MarkdownProps) {
  // Pre-process: rewrite [^N] markers into anchor links so ReactMarkdown renders them
  // as clickable. Out-of-range markers are stripped to avoid dead links.
  const processed =
    citationCount && citationCount > 0
      ? content.replace(/\[\^(\d+)\]/g, (_match, num) => {
          const n = parseInt(num, 10);
          if (n < 1 || n > citationCount) return ""; // drop invalid markers silently
          return `[<sup>${n}</sup>](#citation-${n})`;
        })
      : content;

  return (
    <div
      className={cn(
        "text-[15px] leading-relaxed text-slate-800 space-y-3",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="my-2">{children}</p>,
          strong: ({ children }) => (
            <strong className="font-semibold text-navy">{children}</strong>
          ),
          em: ({ children }) => <em className="italic">{children}</em>,
          ul: ({ children }) => (
            <ul className="list-disc list-outside ml-5 space-y-1.5 my-2">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside ml-5 space-y-1.5 my-2">
              {children}
            </ol>
          ),
          li: ({ children }) => <li className="pl-1">{children}</li>,
          h1: ({ children }) => (
            <h3 className="font-serif font-bold text-xl text-navy mt-4 mb-2">
              {children}
            </h3>
          ),
          h2: ({ children }) => (
            <h3 className="font-serif font-bold text-lg text-navy mt-3 mb-1.5">
              {children}
            </h3>
          ),
          h3: ({ children }) => (
            <h4 className="font-semibold text-navy mt-3 mb-1">{children}</h4>
          ),
          code: ({ children }) => (
            <code className="bg-slate-100 px-1.5 py-0.5 rounded text-[13px] font-mono text-navy">
              {children}
            </code>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-gold-soft pl-4 italic text-slate-600 my-3">
              {children}
            </blockquote>
          ),
          a: ({ href, children }) => {
            if (href?.startsWith("#citation-")) {
              return <CitationAnchor href={href}>{children}</CitationAnchor>;
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-navy underline hover:text-gold-dark"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}