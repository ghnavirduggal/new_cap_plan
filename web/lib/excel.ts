export async function parseExcelFile(file: File): Promise<Record<string, any>[]> {
  const mod = await import("tabularjs");
  const parserMaybe = (mod as any)?.default ?? (mod as any);
  const parser =
    typeof parserMaybe === "function"
      ? parserMaybe
      : typeof parserMaybe?.default === "function"
        ? parserMaybe.default
        : null;
  if (!parser) {
    throw new Error("Excel parser module is invalid.");
  }

  let result: any;
  try {
    result = await parser(file);
  } catch {
    // Some environments/files parse reliably only from raw bytes.
    const buf = await file.arrayBuffer();
    result = await parser(new Uint8Array(buf));
  }

  const worksheet = result?.worksheets?.[0] ?? result?.worksheet ?? result?.sheet ?? null;
  const data: any[][] = Array.isArray(worksheet?.data) ? worksheet.data : [];
  if (!data.length) return [];

  const headers = (data[0] || []).map((header) => String(header ?? "").trim());
  const rows: Array<Record<string, any>> = [];
  for (const row of data.slice(1)) {
    const rowObj: Record<string, any> = {};
    headers.forEach((header, idx) => {
      if (!header) return;
      const value = Array.isArray(row) ? row[idx] : undefined;
      rowObj[header] = value ?? "";
    });
    if (Object.values(rowObj).some((val) => String(val ?? "").trim() !== "")) {
      rows.push(rowObj);
    }
  }

  return rows;
}
