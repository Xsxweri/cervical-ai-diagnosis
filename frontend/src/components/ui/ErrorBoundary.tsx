import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message?: string;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(err: unknown): State {
    return { hasError: true, message: err instanceof Error ? err.message : '未知错误' };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
          <div className="card p-8 max-w-md w-full text-center animate-fade-in">
            <div className="text-4xl mb-3">⚠️</div>
            <h1 className="font-semibold text-gray-900 mb-2">页面发生错误</h1>
            <p className="text-sm text-gray-500 mb-5 break-all">{this.state.message}</p>
            <button
              className="btn-primary"
              onClick={() => {
                this.setState({ hasError: false });
                window.location.reload();
              }}
            >
              刷新重试
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}