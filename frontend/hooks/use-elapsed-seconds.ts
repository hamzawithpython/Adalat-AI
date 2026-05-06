"use client";

import { useEffect, useState } from "react";

export function useElapsedSeconds(startedAt: number | null) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (startedAt === null) {
      setSeconds(0);
      return;
    }
    const tick = () => setSeconds(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const id = setInterval(tick, 250);
    return () => clearInterval(id);
  }, [startedAt]);

  return seconds;
}
