import { apiPostForm } from "./api";

async function parseExcelViaApi(file: File): Promise<Record<string, any>[]> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  const res = await apiPostForm<{ rows?: Record<string, any>[] }>("/api/forecast/ingest/original", formData);
  return Array.isArray(res?.rows) ? res.rows : [];
}

export async function parseExcelFile(file: File): Promise<Record<string, any>[]> {
  let parser: ((input: any) => Promise<any>) | null = null;
  let importErr: any = null;
  try {
    const mod = await import("tabularjs");
    const parserMaybe = (mod as any)?.default ?? (mod as any);
    parser =
      typeof parserMaybe === "function"
        ? parserMaybe
        : typeof parserMaybe?.default === "function"
          ? parserMaybe.default
          : null;
  } catch (err) {
    importErr = err;
  }
  if (!parser) {
    try {
      return await parseExcelViaApi(file);
    } catch (apiErr: any) {
      const modMsg = importErr?.message ? String(importErr.message) : "Excel parser module is invalid.";
      const apiMsg = apiErr?.message ? String(apiErr.message) : "server parser failed";
      throw new Error(`${modMsg}; ${apiMsg}`);
    }
  }

  let result: any;
  let parseErrA: any = null;
  let parseErrB: any = null;
  try {
    result = await parser(file);
  } catch (err) {
    parseErrA = err;
    try {
      // Some environments/files parse reliably only from raw bytes.
      const buf = await file.arrayBuffer();
      result = await parser(new Uint8Array(buf));
    } catch (err2) {
      parseErrB = err2;
      // Final fallback: server-side parser (pandas) for browser/runtime compatibility.
      const rows = await parseExcelViaApi(file);
      if (rows.length) return rows;
      const msgA = parseErrA?.message ? String(parseErrA.message) : "client parser failed";
      const msgB = parseErrB?.message ? String(parseErrB.message) : "byte parser failed";
      throw new Error(`${msgA}; ${msgB}`);
    }
  }

  const worksheet = result?.worksheets?.[0] ?? result?.worksheet ?? result?.sheet ?? null;
  const data: any[][] = Array.isArray(worksheet?.data) ? worksheet.data : [];
  if (!data.length) {
    // If client parser returned no worksheet rows, try server-side parsing before returning empty.
    return parseExcelViaApi(file);
  }

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
