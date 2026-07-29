import type { Metadata } from "next";
import "./globals.css";

// System font stack, no web font fetch, so the build and the app stay offline.
export const metadata: Metadata = {
  title: "Chess Review",
  description: "Review a chess game with a time-aware second opinion, all on your machine.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
