interface IconProps {
  size?: number;
  className?: string;
}

export function ScalesIcon({ size = 32, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={className}>
      <line x1="16" y1="5" x2="16" y2="27" stroke="currentColor" strokeWidth="1.5" />
      <line x1="6" y1="9" x2="26" y2="9" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="16" cy="5" r="1.5" className="fill-gold" />
      <path d="M 4 14 L 6 9 L 8 14 Z" stroke="currentColor" strokeWidth="1.2" fill="none" />
      <path d="M 24 14 L 26 9 L 28 14 Z" stroke="currentColor" strokeWidth="1.2" fill="none" />
      <line x1="12" y1="27" x2="20" y2="27" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function GavelIcon({ size = 32, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={className}>
      <rect x="14" y="4" width="14" height="6" rx="1" transform="rotate(45 21 7)" stroke="currentColor" strokeWidth="1.5" className="fill-gold/15" />
      <line x1="11" y1="14" x2="4" y2="21" stroke="currentColor" strokeWidth="1.8" />
      <rect x="3" y="22" width="14" height="3" rx="1" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function BookIcon({ size = 32, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={className}>
      <rect x="6" y="5" width="20" height="22" rx="1" stroke="currentColor" strokeWidth="1.5" />
      <line x1="10" y1="11" x2="22" y2="11" stroke="currentColor" strokeWidth="1.2" />
      <line x1="10" y1="15" x2="22" y2="15" stroke="currentColor" strokeWidth="1.2" />
      <line x1="10" y1="19" x2="18" y2="19" stroke="currentColor" strokeWidth="1.2" />
      <rect x="20" y="20" width="6" height="3" className="fill-gold" />
    </svg>
  );
}

export function GlobeIcon({ size = 32, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={className}>
      <circle cx="16" cy="16" r="11" stroke="currentColor" strokeWidth="1.5" />
      <ellipse cx="16" cy="16" rx="5" ry="11" stroke="currentColor" strokeWidth="1.2" />
      <line x1="5" y1="16" x2="27" y2="16" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="16" cy="16" r="2" className="fill-gold" />
    </svg>
  );
}

export function ShieldIcon({ size = 32, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" className={className}>
      <path d="M16 4 L26 8 L26 17 C26 22 21 26 16 28 C11 26 6 22 6 17 L6 8 Z" stroke="currentColor" strokeWidth="1.5" className="fill-gold/10" />
      <path d="M11 16 L15 19 L21 12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="stroke-gold" />
    </svg>
  );
}