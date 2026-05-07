"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Flag } from "@/components/brand/flag";
import { GavelIcon } from "@/components/icons/legal-icons";

export function HeroAnswerPreview() {
  return (
    <div className="relative">
      <div className="absolute -inset-4 bg-gradient-to-br from-gold-faint to-transparent rounded-2xl -z-10" />
      <Card padding="lg" className="shadow-brand-lg">
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <Flag code="PK" size={28} />
          <Badge tone="navy">PK</Badge>
          <Badge tone="goldSoft">ROMAN-URDU</Badge>
          <div className="ml-auto inline-flex items-center gap-1.5 text-[11px] font-mono">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-green-700 font-semibold">94% CONFIDENCE</span>
          </div>
        </div>

        <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark mb-2">
          Answer
        </div>
        <p className="text-[14px] leading-relaxed text-slate-700 mb-4">
          <span className="font-semibold text-navy">Sindh Rented Premises Ordinance 1979</span>{" "}
          ke <span className="font-semibold text-navy">Section 13</span> ke mutabiq, landlord
          ko deposit 30 din ke andar wapas karna hai. Agar nahi karta, to aap Rent Controller
          ke paas application kar sakte hain.
        </p>

        <div className="space-y-2 pt-3 border-t border-slate-100">
          <div className="text-[10px] font-mono uppercase tracking-widest text-gold-dark">
            Illustrative Judgment
          </div>
          <div className="flex items-center gap-2.5 p-2.5 rounded-md bg-slate-50 border border-slate-200">
            <div className="w-7 h-7 rounded-md bg-gold-faint border border-gold-soft flex items-center justify-center text-navy">
              <GavelIcon size={14} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[13px] font-serif font-bold text-navy truncate">
                Karachi v. Rashid
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                2018 SCMR 1142 · Supreme Court of Pakistan
              </div>
            </div>
            <span className="text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border bg-green-50 text-green-700 border-green-200">
              Allowed
            </span>
          </div>
        </div>
      </Card>
    </div>
  );
}