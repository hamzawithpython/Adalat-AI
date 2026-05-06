import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes, ReactNode } from "react";

interface BtnProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "gold" | "ghost" | "outline";
  size?: "sm" | "md" | "lg";
  icon?: ReactNode;
  iconRight?: ReactNode;
}

const sizes = {
  sm: "px-3.5 py-2 text-[13px] gap-1.5",
  md: "px-5 py-2.5 text-sm gap-2",
  lg: "px-7 py-3.5 text-[15px] gap-2.5",
};

const variants = {
  primary: "bg-navy text-white border-navy hover:bg-navy-dark",
  gold: "bg-gold text-navy-dark border-gold hover:bg-gold-dark",
  ghost: "bg-transparent text-navy border-slate-200 hover:bg-slate-50",
  outline: "bg-white text-navy border-navy hover:bg-navy/5",
};

export function Btn({
  children,
  variant = "primary",
  size = "md",
  icon,
  iconRight,
  className,
  ...props
}: BtnProps) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center font-medium rounded-lg border transition-colors whitespace-nowrap tracking-tight cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed",
        sizes[size],
        variants[variant],
        className
      )}
    >
      {icon}
      <span>{children}</span>
      {iconRight}
    </button>
  );
}