"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import { Wordmark } from "@/components/brand/wordmark";
import { Flag } from "@/components/brand/flag";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Btn } from "@/components/ui/btn";
import { ScalesIcon, BookIcon, ShieldIcon } from "@/components/icons/legal-icons";

export default function Home() {
  const [status, setStatus] = useState<string>("checking...");

  useEffect(() => {
    getHealth()
      .then((data) => setStatus(`API OK · v${data.version}`))
      .catch((err) => setStatus(`Error: ${err.message}`));
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 p-12">
      <div className="max-w-4xl mx-auto space-y-12">
        {/* Wordmark sizes */}
        <section>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 mb-4">Wordmark</h2>
          <div className="flex items-baseline gap-8">
            <Wordmark size="sm" />
            <Wordmark size="md" />
            <Wordmark size="lg" />
            <Wordmark size="xl" />
          </div>
        </section>

        {/* Badges */}
        <section>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 mb-4">Badges</h2>
          <div className="flex flex-wrap gap-3">
            <Badge tone="navy">Navy</Badge>
            <Badge tone="gold">Gold</Badge>
            <Badge tone="goldSoft">
              <span className="w-1.5 h-1.5 rounded-full bg-gold" /> Beta · PK / UK / DE
            </Badge>
            <Badge tone="success">High confidence</Badge>
            <Badge tone="warning">Medium</Badge>
            <Badge tone="error">Low</Badge>
            <Badge tone="outline">Outline</Badge>
          </div>
        </section>

        {/* Flags */}
        <section>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 mb-4">Flags</h2>
          <div className="flex items-center gap-4">
            <Flag code="PK" /> <Flag code="UK" /> <Flag code="DE" />
          </div>
        </section>

        {/* Icons */}
        <section>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 mb-4">Icons</h2>
          <div className="flex items-center gap-6 text-navy">
            <ScalesIcon size={40} />
            <BookIcon size={40} />
            <ShieldIcon size={40} />
          </div>
        </section>

        {/* Buttons */}
        <section>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 mb-4">Buttons</h2>
          <div className="flex flex-wrap gap-3">
            <Btn variant="primary">Primary</Btn>
            <Btn variant="gold">Gold</Btn>
            <Btn variant="outline">Outline</Btn>
            <Btn variant="ghost">Ghost</Btn>
            <Btn variant="primary" iconRight={<span>→</span>}>
              Try Now
            </Btn>
          </div>
        </section>

        {/* Card with API status */}
        <section>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 mb-4">Card + API</h2>
          <Card padding="lg" className="max-w-md">
            <div className="flex items-center justify-between mb-4">
              <Wordmark size="md" />
              <Badge tone="goldSoft">{status}</Badge>
            </div>
            <p className="text-sm text-slate-600">
              All shared primitives wired up. Ready to build the chat interface.
            </p>
          </Card>
        </section>
      </div>
    </main>
  );
}