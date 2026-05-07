"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

interface MarkdownProps {
  content: string;
  className?: string;
}

export function Markdown({ content, className }: MarkdownProps) {
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
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" className="text-navy underline hover:text-gold-dark">
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}