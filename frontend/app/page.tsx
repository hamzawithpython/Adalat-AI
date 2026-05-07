"use client";

import Link from "next/link";
import { Wordmark } from "@/components/brand/wordmark";
import { Btn } from "@/components/ui/btn";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Flag } from "@/components/brand/flag";
import {
  ScalesIcon,
  BookIcon,
  ShieldIcon,
  GlobeIcon,
  GavelIcon,
} from "@/components/icons/legal-icons";
import { HeroAnswerPreview } from "@/components/landing/hero-answer-preview";

export default function LandingPage() {
  return (
    <div className="bg-white text-slate-800 font-sans">
      <NavBar />
      <Hero />
      <HowItWorks />
      <Jurisdictions />
      <WhyAdalat />
      <SampleQueries />
      <FinalCTA />
      <Footer />
    </div>
  );
}

function NavBar() {
  return (
    <nav className="sticky top-0 z-40 bg-white/90 backdrop-blur border-b border-slate-100">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 lg:px-16 py-5">
        <Wordmark size="md" />
        <div className="hidden md:flex items-center gap-9 text-sm text-slate-700">
          <a href="#how-it-works" className="hover:text-navy">How it works</a>
          <a href="#jurisdictions" className="hover:text-navy">Jurisdictions</a>
          <a href="#why" className="hover:text-navy">Why Adalat</a>
        </div>
        <Link href="/chat">
          <Btn variant="primary" size="sm" iconRight={<span>→</span>}>
            Try Now
          </Btn>
        </Link>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <section className="px-6 lg:px-16 pt-16 pb-24 lg:pt-24 lg:pb-32 bg-gradient-to-b from-white to-slate-50">
      <div className="max-w-7xl mx-auto grid lg:grid-cols-[1.15fr_1fr] gap-12 lg:gap-16 items-center">
        <div>
          <Badge tone="goldSoft" className="mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-gold" />
            BETA · PK / UK / DE
          </Badge>
          <h1 className="font-serif font-bold text-[40px] sm:text-5xl lg:text-6xl tracking-tight text-navy leading-[1.05] mb-6">
            Your Rights.
            <br />
            <span className="italic">In Your Language.</span>
          </h1>
          <p className="text-lg text-slate-600 max-w-lg mb-10 leading-relaxed">
            Ask a legal question in Roman-Urdu, English, or German. Get a structured
            answer with article-level citations from Pakistani, UK, and German law.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/chat">
              <Btn variant="primary" size="lg" iconRight={<span>→</span>}>
                Ask Adalat
              </Btn>
            </Link>
            <a href="#how-it-works">
              <Btn variant="ghost" size="lg">How it works</Btn>
            </a>
          </div>
          <div className="mt-10 flex items-center gap-6 text-xs font-mono text-slate-500">
            <Stat value="47" label="Legal docs" />
            <Stat value="8,322" label="Vector chunks" />
            <Stat value="3" label="Jurisdictions" />
          </div>
        </div>

        <HeroAnswerPreview />
      </div>
    </section>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="text-2xl font-serif font-bold text-navy">{value}</div>
      <div className="uppercase tracking-widest text-[10px] mt-0.5">{label}</div>
    </div>
  );
}

function HowItWorks() {
  const steps = [
    { n: "01", icon: <GlobeIcon size={28} />, t: "Ask in your language", d: "Roman-Urdu, English, or German. Long descriptions welcome — every word matters." },
    { n: "02", icon: <BookIcon size={28} />, t: "We retrieve the law", d: "Statutes, ordinances, and acts are searched and ranked by relevance." },
    { n: "03", icon: <ShieldIcon size={28} />, t: "You get a sourced answer", d: "Structured response with rights, deadlines, and exact article citations." },
  ];

  return (
    <section id="how-it-works" className="px-6 lg:px-16 py-24 lg:py-28 bg-white">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-14 lg:mb-16">
          <div className="text-[11px] font-mono font-semibold text-gold uppercase tracking-[0.2em] mb-3">
            How it works
          </div>
          <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-navy tracking-tight mb-4">
            Three steps. No legalese.
          </h2>
          <p className="text-base text-slate-600 max-w-xl mx-auto">
            From question to sourced answer in under 60 seconds.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-5">
          {steps.map((s) => (
            <Card key={s.n} padding="lg" hoverable>
              <div className="flex items-start justify-between mb-6">
                <div className="w-14 h-14 rounded-lg bg-gold-faint flex items-center justify-center text-navy">
                  {s.icon}
                </div>
                <span className="font-mono text-xl font-bold text-slate-200">{s.n}</span>
              </div>
              <h3 className="font-serif text-xl font-bold text-navy mb-2">{s.t}</h3>
              <p className="text-[15px] text-slate-600 leading-relaxed">{s.d}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}

function Jurisdictions() {
  const items = [
    { code: "PK" as const, name: "Pakistan", docs: "Constitution, PPC, CrPC, Tenancy Acts (Punjab/Sindh/KP/ICT), Family Laws", count: "20+ docs" },
    { code: "UK" as const, name: "United Kingdom", docs: "Tenant Fees Act, Housing Acts, Consumer Rights Act, Employment Rights, Equality Act", count: "12+ docs" },
    { code: "DE" as const, name: "Germany", docs: "BGB, Mietrechtsreformgesetz, Betriebskostenverordnung, UWG", count: "10+ docs" },
  ];

  return (
    <section id="jurisdictions" className="px-6 lg:px-16 py-24 lg:py-28 bg-slate-50">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-14 lg:mb-16">
          <div className="text-[11px] font-mono font-semibold text-gold uppercase tracking-[0.2em] mb-3">
            Jurisdictions
          </div>
          <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-navy tracking-tight mb-4">
            Three legal systems.
            <br />
            One assistant.
          </h2>
        </div>
        <div className="grid md:grid-cols-3 gap-5">
          {items.map((j) => (
            <Card key={j.code} padding="lg">
              <div className="flex items-center gap-3 mb-4">
                <Flag code={j.code} size={36} />
                <div>
                  <h3 className="font-serif text-xl font-bold text-navy">{j.name}</h3>
                  <div className="text-[11px] font-mono text-slate-500">{j.count}</div>
                </div>
              </div>
              <p className="text-[14px] text-slate-600 leading-relaxed">{j.docs}</p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}

function WhyAdalat() {
  const features = [
    { icon: <ScalesIcon size={24} />, t: "Article-level citations", d: "Every answer points to the exact statute, section, and page. No hand-waving." },
    { icon: <GlobeIcon size={24} />, t: "Multilingual by design", d: "Responses match the language you write in. Roman-Urdu in, Roman-Urdu out." },
    { icon: <GavelIcon size={24} />, t: "Structured rights extraction", d: "Each answer breaks down rights with legal basis, deadlines, and recourse." },
    { icon: <ShieldIcon size={24} />, t: "Honest disclaimers", d: "Illustrative judgments are clearly labelled. Your data isn't sold or trained on." },
  ];

  return (
    <section id="why" className="px-6 lg:px-16 py-24 lg:py-28 bg-white">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-[1fr_1.4fr] gap-12 lg:gap-20 items-start">
          <div>
            <div className="text-[11px] font-mono font-semibold text-gold uppercase tracking-[0.2em] mb-3">
              Why Adalat
            </div>
            <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-navy tracking-tight mb-5 leading-tight">
              Built for people who can't afford a first consultation.
            </h2>
            <p className="text-slate-600 leading-relaxed">
              Most legal AI tools are toys, or charge by the seat. Adalat-AI grounds every
              answer in real statutes, in the language you actually speak, with citations
              you can verify. Free. No login. No bullshit.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            {features.map((f) => (
              <Card key={f.t} padding="md">
                <div className="w-10 h-10 rounded-md bg-gold-faint flex items-center justify-center text-navy mb-3">
                  {f.icon}
                </div>
                <h4 className="font-serif text-lg font-bold text-navy mb-1.5">{f.t}</h4>
                <p className="text-[14px] text-slate-600 leading-relaxed">{f.d}</p>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function SampleQueries() {
  const samples = [
    { code: "PK" as const, lang: "Roman-Urdu", q: "Mera landlord deposit wapas nahi de raha, kya karoon?" },
    { code: "UK" as const, lang: "English", q: "My landlord won't return my £1,400 deposit. What are my rights?" },
    { code: "DE" as const, lang: "Deutsch", q: "Mein Vermieter gibt meine Kaution nicht zurück. Was tun?" },
  ];

  return (
    <section className="px-6 lg:px-16 py-24 lg:py-28 bg-slate-50">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <div className="text-[11px] font-mono font-semibold text-gold uppercase tracking-[0.2em] mb-3">
            Try it
          </div>
          <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-navy tracking-tight mb-4">
            Sample queries.
          </h2>
          <p className="text-slate-600">Click any query to try it on the live app.</p>
        </div>
        <div className="space-y-3">
          {samples.map((s) => (
            <Link
              key={s.code}
              href={`/chat?q=${encodeURIComponent(s.q)}`}
              className="flex items-center gap-4 px-5 py-4 bg-white border border-slate-200 rounded-lg hover:border-navy hover:shadow-brand transition-all group"
            >
              <Flag code={s.code} size={32} />
              <div className="flex-1 min-w-0">
                <div className="text-[10px] font-mono uppercase tracking-widest text-slate-400 mb-1">
                  {s.lang}
                </div>
                <div className="text-[15px] text-slate-700 group-hover:text-navy">
                  {s.q}
                </div>
              </div>
              <span className="text-slate-300 group-hover:text-navy transition-colors text-xl">→</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function FinalCTA() {
  return (
    <section className="px-6 lg:px-16 py-20 lg:py-24 bg-navy text-white">
      <div className="max-w-3xl mx-auto text-center">
        <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
          Know your rights. In <span className="italic text-gold">your</span> language.
        </h2>
        <p className="text-slate-300 mb-8 max-w-xl mx-auto">
          Free. No login required. Three jurisdictions, three languages.
        </p>
        <Link href="/chat">
          <Btn variant="gold" size="lg" iconRight={<span>→</span>}>
            Start asking
          </Btn>
        </Link>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="px-6 lg:px-16 py-10 bg-slate-900 text-slate-400 text-sm">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Wordmark size="sm" className="text-white" />
          <span className="text-slate-600">·</span>
          <span className="text-xs">Built as a portfolio capstone.</span>
        </div>
        <div className="text-xs font-mono">
          v1.0 · Not a substitute for legal advice
        </div>
      </div>
    </footer>
  );
}