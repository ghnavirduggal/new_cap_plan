export async function parseExcelFile(file: File): Promise<Record<string, any>[]> {
  const { default: tabularjs } = await import("tabularjs");
  const result = await tabularjs(file);
  const worksheet = result?.worksheets?.[0];
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
