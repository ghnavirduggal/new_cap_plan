/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const forecastBase = process.env.NEXT_PUBLIC_FORECAST_URL || "http://localhost:8082";
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    return [
      { source: "/api/planning/:path*", destination: `${forecastBase}/api/planning/:path*` },
      { source: "/api/forecast/:path*", destination: `${forecastBase}/api/forecast/:path*` },
      { source: "/api/uploads/:path*", destination: `${forecastBase}/api/uploads/:path*` },
      // Auth/session, profile and user-directory endpoints are served by the
      // forecast service (FastAPI), not the Go API — route them there so token
      // minting and profile save don't 500 against the wrong backend.
      { source: "/api/auth/:path*", destination: `${forecastBase}/api/auth/:path*` },
      { source: "/api/user", destination: `${forecastBase}/api/user` },
      { source: "/api/users/:path*", destination: `${forecastBase}/api/users/:path*` },
      { source: "/api/:path*", destination: `${apiBase}/api/:path*` }
    ];
  }
};

module.exports = nextConfig;
