# AI-Agent 宫颈病理诊断系统 · 前端

React 18 + TypeScript + Vite + Tailwind CSS。提供仪表盘、智能诊断（Agent）、快速分析、诊断历史四个页面。

## 本地开发

```bash
npm install
npm run dev
```

默认连接 `http://localhost:8000`（HTTP）与 `ws://localhost:8000`（WebSocket）。

## 环境变量

复制 `.env.example` 为 `.env` 并按部署环境填写：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `VITE_API_BASE` | 后端 HTTP 基础地址，如 `https://api.example.com` | `http://localhost:8000` |
| `VITE_WS_BASE` | 后端 WebSocket 基础地址，如 `wss://api.example.com` | `ws://localhost:8000` |

> 生产环境必须把 `VITE_WS_BASE` 设为 `wss://`，否则 HTTPS 页面会被浏览器判定为混合内容并阻断 WebSocket。
> 变量在 **构建时** 内联进产物，修改后需要重新 `npm run build`。

## 构建与类型检查

```bash
npm run typecheck   # tsc -b：会真正检查 referenced projects
npm run build       # tsc -b && vite build
npm run lint        # eslint .
```

`dist/` 为构建产物，**部署前必须重新构建**，不要直接使用仓库中已有的旧产物。

## 鉴权与安全说明

- 登录 token 保存在 `localStorage`（键名 `cervical-auth-token`），HTTP 请求走 `Authorization: Bearer`。
- 401 响应会自动清除 token 并跳转 `/login`。
- WebSocket 通过 `Sec-WebSocket-Protocol: bearer, <JWT>` 子协议传递凭证，不使用 `?token=`（避免凭证进入访问日志与浏览器历史）。后端必须回选 `bearer` 子协议，否则浏览器判定握手失败（1006）。
- `localStorage` 可被 XSS 读取，生产环境建议配置 CSP，或改为 httpOnly Cookie 方案。

## 风险展示约定

- `risk_level` 有 `low | medium | high | emergency | unreadable` 五档；
  `unreadable` 表示"未检出可判读细胞"，UI 用灰色徽章显式提示，**不会显示成低风险**。
- `risk_scope` 区分 `cell`（细胞级提示）与 `case`（病例级结论），两者在徽章上可区分。
- 不确定性颜色档位优先使用后端返回的 `uncertainty_thresholds`，写死的默认值仅作兜底。

## 目录结构

```
src/
  components/ui/   通用组件（徽章、表格、弹窗、Toast…）
  components/layout/
  hooks/           useDiagnosis：智能诊断流程状态机
  libts/           api（HTTP/WS）、format（风险与格式化）、storage（本地历史）
  pages/           仪表盘 / 智能诊断 / 快速分析 / 诊断历史 / 登录 / 404
  types/           与后端响应对应的类型定义
```

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])

```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])

```
