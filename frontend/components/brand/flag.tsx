import type { Jurisdiction } from "@/types/legal";

interface FlagProps {
  code: Jurisdiction;
  size?: number;
}

export function Flag({ code, size = 36 }: FlagProps) {
  const height = size * 0.66;

  if (code === "PK") {
    return (
      <svg width={size} height={height} viewBox="0 0 36 24" className="rounded-sm shadow-sm">
        <rect width="36" height="24" fill="#01411C" />
        <rect width="9" height="24" fill="#FFFFFF" />
        <circle cx="22" cy="12" r="5.5" fill="#01411C" />
        <circle cx="23.5" cy="11" r="5" fill="#FFFFFF" />
        <polygon
          points="26,9 26.7,11 28.7,11 27.1,12.2 27.7,14.2 26,13 24.3,14.2 24.9,12.2 23.3,11 25.3,11"
          fill="#FFFFFF"
        />
      </svg>
    );
  }

  if (code === "UK") {
    return (
      <svg width={size} height={height} viewBox="0 0 36 24" className="rounded-sm shadow-sm">
        <rect width="36" height="24" fill="#012169" />
        <path d="M 0 0 L 36 24 M 36 0 L 0 24" stroke="#FFFFFF" strokeWidth="3" />
        <path d="M 0 0 L 36 24 M 36 0 L 0 24" stroke="#C8102E" strokeWidth="1.5" />
        <rect x="15" width="6" height="24" fill="#FFFFFF" />
        <rect y="9" width="36" height="6" fill="#FFFFFF" />
        <rect x="16.5" width="3" height="24" fill="#C8102E" />
        <rect y="10.5" width="36" height="3" fill="#C8102E" />
      </svg>
    );
  }

  if (code === "DE") {
    return (
      <svg width={size} height={height} viewBox="0 0 36 24" className="rounded-sm shadow-sm">
        <rect width="36" height="8" fill="#000000" />
        <rect y="8" width="36" height="8" fill="#DD0000" />
        <rect y="16" width="36" height="8" fill="#FFCE00" />
      </svg>
    );
  }

  return null;
}