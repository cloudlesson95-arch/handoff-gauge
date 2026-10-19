import type { NextConfig } from "next";

// The Python API (FastAPI, api/index.py) is served under /api/py/*.
// In development it runs as a local uvicorn on port 8000; on Vercel it is the
// Python function at /api, which receives the original /api/py/... path.
const pythonApi =
  process.env.NODE_ENV === "development"
    ? "http://127.0.0.1:8000/api/py/:path*"
    : "/api/";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [{ source: "/api/py/:path*", destination: pythonApi }];
  },
};

export default nextConfig;
