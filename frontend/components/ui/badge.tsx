import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

type BadgeTone =
  | "default"
  | "navy"
  | "navySoft"
  | "gold"
  | "goldSoft"
  | "success"
  | "warning"
  | "error"
  | "outline";

interface BadgeProps {
  children: ReactNode;
  tone?: BadgeTone;
  icon?: ReactNode;
  className?: string;
}

const toneStyles: Record<BadgeTone, string> = {
  default: "bg-slate-100 text-slate-700 border-slate-200",
  navy: "bg-navy text-white border-navy",
  navySoft: "bg-navy/10 text-navy border-transparent",
  gold: "bg-gold text-navy-dark border-gold",
  goldSoft: "bg-gold-faint text-gold-dark border-gold-soft",
  success: "bg-green-100 text-green-700 border-transparent",
  warning: "bg-amber-100 text-amber-800 border-transparent",
  error: "bg-red-100 text-red-700 border-transparent",
  outline: "bg-white text-navy border-navy",
};

export function Badge({ children, tone = "default", icon, className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider border",
        toneStyles[tone],
        className
      )}
    >
      {icon}
      {children}
    </span>
  );
}