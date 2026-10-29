#!/bin/sh
# 每日备份 + 保留 N 天，随 pgbackup 容器运行
set -eu
KEEP="${BACKUP_KEEP_DAYS:-14}"
mkdir -p /backup

while true; do
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  FILE="/backup/cervix_${TS}.dump"

  if pg_dump -h postgres -U cervix -d cervix -Fc -f "$FILE"; then
    echo "[backup] ok $FILE ($(du -h "$FILE" | cut -f1))"
  else
    echo "[backup] FAILED $TS" >&2
    rm -f "$FILE"
  fi

  # 删除超期备份（只删本脚本生成的文件，避免误删）
  find /backup -maxdepth 1 -name 'cervix_*.dump' -type f -mtime "+${KEEP}" -print -delete

  sleep 86400
done