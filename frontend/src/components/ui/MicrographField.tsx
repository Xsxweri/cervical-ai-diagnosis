/**
 * 合成的宫颈细胞学视野（H&E 色调）。
 *
 * 刻意不使用真实病例图像：这里只作为"示意视野"，不承载任何临床信息，
 * 也不存在去标识化不彻底的风险。对焦环用 CSS 叠加而非烘进图片，
 * 以便与焦点细胞核精确对齐、并单独做一次性动效。
 */

interface CellSpec {
  left: string;
  top: string;
  size: number;
  nucleus: number;
  opacity: number;
}

const CELLS: CellSpec[] = [
  // 焦点细胞：与下方对焦环共用 50% / 44% 定位
  { left: '50%', top: '44%', size: 186, nucleus: 54, opacity: 1 },
  { left: '16%', top: '18%', size: 128, nucleus: 36, opacity: 0.72 },
  { left: '79%', top: '24%', size: 104, nucleus: 30, opacity: 0.58 },
  { left: '24%', top: '70%', size: 150, nucleus: 42, opacity: 0.66 },
  { left: '70%', top: '77%', size: 132, nucleus: 38, opacity: 0.7 },
  { left: '89%', top: '52%', size: 88, nucleus: 26, opacity: 0.48 },
  { left: '6%', top: '47%', size: 96, nucleus: 28, opacity: 0.44 },
  { left: '43%', top: '89%', size: 110, nucleus: 32, opacity: 0.4 },
  { left: '58%', top: '9%', size: 92, nucleus: 26, opacity: 0.36 },
];

export default function MicrographField() {
  return (
    <>
      {/* 视野本体：载入时由失焦收束到清晰 */}
      <div className="animate-focus-in absolute inset-0">
        {/* 底色：淡伊红 */}
        <div
          className="absolute inset-0"
          style={{
            background:
              'radial-gradient(120% 90% at 50% 40%, #F0E7E4 0%, #E4D8D5 55%, #D6C9C6 100%)',
          }}
        />

        {/* 细胞：核（苏木精）+ 胞浆（伊红），低饱和 */}
        {CELLS.map((cell, i) => {
          const nucleusPct = (cell.nucleus / cell.size) * 50;
          return (
            <span
              key={i}
              className="absolute rounded-full"
              style={{
                left: cell.left,
                top: cell.top,
                width: cell.size,
                height: cell.size,
                transform: 'translate(-50%, -50%)',
                opacity: cell.opacity,
                filter: 'blur(1.2px)',
                background: `radial-gradient(circle at 50% 50%,
                  rgba(74,84,116,0.46) 0 ${nucleusPct}%,
                  rgba(206,188,186,0.62) ${nucleusPct + 1}% 74%,
                  rgba(222,208,205,0.30) 75% 100%)`,
              }}
            />
          );
        })}

        {/* 上下压暗，保证白色字标与标本说明可读 */}
        <div
          className="absolute inset-0"
          style={{
            background:
              'linear-gradient(to bottom, rgba(18,23,26,0.42) 0%, rgba(18,23,26,0.05) 30%, rgba(18,23,26,0.10) 62%, rgba(18,23,26,0.55) 100%)',
          }}
        />
        {/* 左侧轻微压暗，向表单一侧过渡 */}
        <div
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(to right, rgba(18,23,26,0.30) 0%, transparent 45%)',
          }}
        />
      </div>

      {/* 对焦环：112px，1px 白 72%，四向刻度 + 中心点，只绘制一次 */}
      <div
        className="pointer-events-none absolute left-1/2 top-[44%] -translate-x-1/2 -translate-y-1/2"
        aria-hidden="true"
      >
        <div className="animate-ring-in relative h-28 w-28">
          <span className="absolute inset-0 rounded-full border border-white/[0.72]" />
          <span className="absolute left-1/2 top-0 h-2 w-px -translate-x-1/2 bg-white/[0.72]" />
          <span className="absolute bottom-0 left-1/2 h-2 w-px -translate-x-1/2 bg-white/[0.72]" />
          <span className="absolute left-0 top-1/2 h-px w-2 -translate-y-1/2 bg-white/[0.72]" />
          <span className="absolute right-0 top-1/2 h-px w-2 -translate-y-1/2 bg-white/[0.72]" />
          <span className="absolute left-1/2 top-1/2 h-[3px] w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/[0.85]" />
        </div>
      </div>
    </>
  );
}