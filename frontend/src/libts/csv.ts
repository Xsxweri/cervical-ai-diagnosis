/** CSV 生成与下载 */

function escapeCell(v: unknown): string {
  if (v == null) return '';
  const s = String(v);
  // 含逗号、引号、换行时必须整体加引号，内部引号翻倍
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(headers: string[], rows: unknown[][]): string {
  const lines = [headers.map(escapeCell).join(',')];
  for (const row of rows) lines.push(row.map(escapeCell).join(','));
  // BOM：否则 Excel 会把 UTF-8 中文识别成乱码
  return '\ufeff' + lines.join('\r\n');
}

export function downloadText(
  filename: string,
  text: string,
  mime = 'text/csv;charset=utf-8',
): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // 立刻 revoke 会让部分浏览器取消下载，延后一拍
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** 文件名安全化：去掉 Windows/POSIX 非法字符 */
export function safeFilename(name: string, fallback = 'export'): string {
  const cleaned = name.replace(/[\\/:*?"<>|\r\n]/g, '_').trim();
  return cleaned || fallback;
}