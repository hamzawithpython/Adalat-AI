import {
  ScalesIcon,
  BookIcon,
  ShieldIcon,
  GavelIcon,
  GlobeIcon,
} from "@/components/icons/legal-icons";
import type { IconHint } from "@/types/legal";

const ICON_MAP = {
  scales: ScalesIcon,
  book: BookIcon,
  shield: ShieldIcon,
  gavel: GavelIcon,
  globe: GlobeIcon,
} as const;

interface SectionIconProps {
  hint?: IconHint | null;
  size?: number;
  className?: string;
}

export function SectionIcon({ hint, size = 22, className }: SectionIconProps) {
  // Default to scales if hint is missing or unknown
  const Icon = (hint && ICON_MAP[hint]) || ScalesIcon;
  return <Icon size={size} className={className} />;
}