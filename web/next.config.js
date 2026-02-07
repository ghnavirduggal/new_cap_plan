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
      { source: "/api/:path*", destination: `${apiBase}/api/:path*` }
    ];
  }
};

module.exports = nextConfig;
