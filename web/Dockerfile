FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
ARG FORECAST_URL=http://forecast-service:8080
ARG API_URL=http://api:8080
ENV FORECAST_URL=${FORECAST_URL}
ENV API_URL=${API_URL}
RUN npm run build

FROM node:20-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
# Run as a non-root user. Use the numeric UID:GID (1000 = the built-in "node"
# user on official node images) rather than the name "node": a numeric USER
# needs no /etc/passwd entry, so the container starts even on bases that don't
# register the node user (avoids "unable to find user node: no matching entries
# in passwd files").
COPY --from=build --chown=1000:1000 /app ./
USER 1000:1000
EXPOSE 3000
# Production server (not the dev server, which leaks source maps / verbose errors
# and applies no production hardening).
CMD ["npm", "run", "start"]
