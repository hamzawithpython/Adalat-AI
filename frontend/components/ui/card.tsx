import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  hoverable?: boolean;
  padding?: "sm" | "md" | "lg";
}

const paddings = {
  sm: "p-4",
  md: "p-6",
  lg: "p-8",
};

export function Card({ children, className, hoverable, padding = "md" }: CardProps) {
  return (
    <div
      className={cn(
        "bg-white border border-slate-200 rounded-lg shadow-brand-sm transition-shadow",
        paddings[padding],
        hoverable && "cursor-pointer hover:shadow-brand-md",
        className
      )}
    >
      {children}
    </div>
  );
}