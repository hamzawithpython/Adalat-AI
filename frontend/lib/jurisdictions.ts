import type { Jurisdiction } from "@/types/legal";

export const JURISDICTIONS: {
  code: Jurisdiction;
  name: string;
  shortName: string;
}[] = [
  { code: "PK", name: "Pakistan", shortName: "PK" },
  { code: "UK", name: "United Kingdom", shortName: "UK" },
  { code: "DE", name: "Germany", shortName: "DE" },
];

export const SAMPLE_QUERIES: {
  code: Jurisdiction;
  query: string;
  language: string;
}[] = [
  {
    code: "PK",
    query: "Mera landlord deposit wapas nahi de raha, kya karoon?",
    language: "Roman-Urdu",
  },
  {
    code: "UK",
    query: "My landlord won't return my £1,400 deposit. What can I do?",
    language: "English",
  },
  {
    code: "DE",
    query: "Mein Vermieter gibt die Kaution nicht zurück.",
    language: "Deutsch",
  },
];