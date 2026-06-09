import { cn } from "@/lib/utils";

interface WordmarkProps {
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
}

const sizes = {
  sm: { text: "text-lg", dot: "w-1 h-1 mx-1", ai: "text-xs ml-0.5" },
  md: { text: "text-2xl", dot: "w-1.5 h-1.5 mx-1.5", ai: "text-sm ml-1" },
  lg: { text: "text-4xl", dot: "w-2 h-2 mx-2", ai: "text-xl ml-1.5" },
  xl: { text: "text-6xl", dot: "w-3 h-3 mx-2.5", ai: "text-3xl ml-2" },
};

export function Wordmark({ size = "md", className }: WordmarkProps) {
  const s = sizes[size];
  return (
    <div
      className={cn(
      "inline-flex items-baseline font-serif font-bold tracking-tight text-navy",
      s.text,
      className
    )}
    >
      Adalat
      <span
        className={cn("inline-block rounded-full bg-gold -translate-y-1", s.dot)}
        aria-hidden="true"
      />
      <span className={cn("font-sans font-semibold tracking-wider", s.ai)}>
        AI
      </span>
    </div>
  );
}