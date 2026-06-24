FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
# Run as the built-in non-root node user.
COPY --from=build --chown=node:node /app ./
USER node
EXPOSE 3000
# Production server (not the dev server, which leaks source maps / verbose errors
# and applies no production hardening).
CMD ["npm", "run", "start"]
