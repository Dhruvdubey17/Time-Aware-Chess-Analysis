import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Export a fully static site (this app is a client-side SPA), so the local
  // backend can serve it same-origin and the launch is one process, no Node at
  // runtime. `next dev` is unaffected.
  output: "export",
  images: { unoptimized: true },
  // Pin the workspace root to this app so a stray lockfile elsewhere on the
  // machine cannot be mistaken for the project root during build.
  turbopack: { root: __dirname },
};

export default nextConfig;
