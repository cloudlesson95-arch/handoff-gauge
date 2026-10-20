"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQueue } from "@/lib/queue";

const LINKS = [
  { href: "/", label: "Triage" },
  { href: "/review", label: "Review queue" },
  { href: "/calibration", label: "Calibration" },
] as const;

export default function Nav() {
  const pathname = usePathname();
  const open = useQueue().filter((item) => !item.resolved).length;
  return (
    <nav className="flex gap-1 text-sm" aria-label="Main">
      {LINKS.map(({ href, label }) => {
        const active = pathname === href;
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`rounded-md px-3 py-1.5 ${
              active ? "bg-wash font-medium text-ink" : "text-ink-2 hover:bg-wash"
            }`}
          >
            {label}
            {href === "/review" && open > 0 && (
              <span className="tabular ml-1.5 rounded-full border border-line px-1.5 text-xs">
                {open}
              </span>
            )}
          </Link>
        );
      })}
    </nav>
  );
}
