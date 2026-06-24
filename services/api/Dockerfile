FROM golang:1.22-alpine AS build
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# No `go mod tidy` at build time: build from the committed go.mod/go.sum only so
# the image is reproducible and can't silently pull altered dependencies.
RUN CGO_ENABLED=0 go build -o /out/api .

FROM alpine:3.20
RUN adduser -D app
USER app
WORKDIR /app
COPY --from=build /out/api /app/api
EXPOSE 8080
ENTRYPOINT ["/app/api"]
