/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // 品牌主色：苏木精蓝整条色阶（原天蓝已替换）
        primary: {
          50: '#F4F6FA',
          100: '#E7EBF3',
          200: '#CBD3E4',
          300: '#A8B4CE',
          400: '#7A8BB0',
          500: '#54668F',
          600: '#2E3B63',
          700: '#263152',
          800: '#1E2741',
          900: '#161E33',
        },
        // 语义色：正常 / 警告 / 异常（保持原有键，避免全站改类名）
        success: {
          50: '#EDF2EE', 100: '#E3EAE5', 200: '#C7D6CC',
          500: '#6E8F79', 600: '#4F7A63', 700: '#3C5D4B',
        },
        warning: {
          50: '#F6F0E2', 100: '#F1E7D2', 200: '#E4D3B2',
          500: '#B39356', 600: '#9A7B3F', 700: '#6E5527',
          800: '#55411E',
        },
        danger: {
          50: '#F8EBE7', 100: '#F4DED8', 200: '#E8C2B7',
          500: '#C4705A', 600: '#B0563F', 700: '#8A4230',
        },
        // 原"AI 紫"退出舞台，映射为中性灰（保留键名，避免改动 8 处引用）
        ai: {
          50: '#F2F5F3', 100: '#E8ECEA', 200: '#D2D9D6',
          500: '#8A9391', 600: '#667076', 700: '#4E565A',
        },

        // ---- H&E 设计令牌 ----
        glass: '#F2F5F3',
        ink: '#12171A',
        hematoxylin: '#2E3B63',
        eosin: '#A64F52',
        muted: '#667076',
        hairline: '#DCE2DF',

        // ---- 低饱和风险色：底色 / 文字（两档分开，保证对比度）----
        risk: {
          low: '#4F7A63',
          lowInk: '#3C5D4B',
          lowBg: '#EDF2EE',
          medium: '#9A7B3F',
          mediumInk: '#6E5527',
          mediumBg: '#F6F0E2',
          high: '#B0563F',
          highInk: '#8A4230',
          highBg: '#F8EBE7',
          emergency: '#8E2F35',
          emergencyBg: '#F6E7E8',
          unreadable: '#6B7478',
          unreadableInk: '#565E62',
          unreadableBg: '#EFF1F2',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"',
          '"PingFang SC"', '"Hiragino Sans GB"', '"Microsoft YaHei"',
          '"Noto Sans SC"', 'sans-serif',
        ],
        // 医疗报告的书卷感：思源宋体优先，回退系统宋体
        serif: [
          '"Noto Serif SC"', '"Source Han Serif SC"',
          '"Songti SC"', '"SimSun"', 'serif',
        ],
      },
      boxShadow: {
        panel: '0 12px 32px -12px rgb(18 23 26 / 0.18), 0 2px 8px -4px rgb(18 23 26 / 0.08)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        // 视野由失焦收束到清晰
        'focus-in': {
          '0%': { filter: 'blur(10px)', transform: 'scale(1.03)', opacity: '0.85' },
          '100%': { filter: 'blur(0px)', transform: 'scale(1)', opacity: '1' },
        },
        // 对焦环淡入并缩放一次（平移由外层容器负责）
        'ring-in': {
          '0%': { opacity: '0', transform: 'scale(0.94)' },
          '60%': { opacity: '1' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        // 按钮底部不确定进度线
        'progress-line': {
          '0%': { transform: 'translateX(-110%)' },
          '100%': { transform: 'translateX(340%)' },
        },
        'drawer-in': {
          '0%': { transform: 'translateX(16px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.3s ease-out',
        'pulse-soft': 'pulse-soft 1.6s ease-in-out infinite',
        'focus-in': 'focus-in 800ms cubic-bezier(.2,.8,.2,1) both',
        'ring-in': 'ring-in 800ms cubic-bezier(.2,.8,.2,1) both',
        'progress-line': 'progress-line 1.1s cubic-bezier(.4,0,.6,1) infinite',
        'drawer-in': 'drawer-in 180ms cubic-bezier(.2,.8,.2,1) both',
      },
    },
  },
  plugins: [],
}