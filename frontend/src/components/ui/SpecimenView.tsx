import { useEffect, useState } from 'react';
import { FileArchive } from 'lucide-react';
import type { CellResult } from '../../types';

interface Props {
  /** 本地 File（刚上传/批量选择时可用） */
  file?: File;
  /** 远程图片地址（会话恢复时由 fetchSessionImage 生成 blob URL） */
  src?: string | null;
  cells?: CellResult[];
  alt: string;
  className?: string;
}

/**
 * 原图 + 检测框。
 *
 * bbox 是原图坐标，因此用 <svg viewBox> 让框与图共享坐标系自动缩放；
 * vectorEffect="non-scaling-stroke" 保证缩略图尺寸下线宽仍然可见。
 */
export default function SpecimenView({ file, src, cells, alt, className = '' }: Props) {
  const [url, setUrl] = useState<string | null>(null);
  const [dim, setDim] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setUrl(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    }
    // src 的生命周期由调用方管理（blob URL 需由其负责 revoke）
    setUrl(src ?? null);
  }, [file, src]);

  useEffect(() => {
    if (!url) {
      setDim(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled) setDim({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.onerror = () => {
      if (!cancelled) setDim(null);
    };
    img.src = url;
    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [url]);

  // 无图（ZIP 结果、原图已清理）：保留同尺寸占位，避免行高跳动
  if (!url) {
    return (
      <div
        className={`flex items-center justify-center bg-glass ${className}`}
        aria-hidden="true"
      >
        <FileArchive size={16} className="text-muted/50" />
      </div>
    );
  }

  return (
    <div className={`overflow-hidden bg-ink/5 ${className}`}>
      {dim ? (
        <svg
          viewBox={`0 0 ${dim.w} ${dim.h}`}
          className="h-full w-full"
          role="img"
          aria-label={alt || '细胞学视野与检测框'}
        >
          <image href={url} width={dim.w} height={dim.h} />
          {(cells ?? []).map((c, i) => (
            <rect
              key={i}
              x={c.bbox[0]}
              y={c.bbox[1]}
              width={Math.max(c.bbox[2] - c.bbox[0], 1)}
              height={Math.max(c.bbox[3] - c.bbox[1], 1)}
              fill="none"
              stroke={c.binary_result.class === 'abnormal' ? '#B0563F' : '#4F7A63'}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      ) : (
        <div className="h-full w-full animate-pulse-soft bg-glass" />
      )}
    </div>
  );
}