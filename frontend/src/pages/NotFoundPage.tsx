import { Link } from 'react-router-dom';
import { Compass } from 'lucide-react';
import EmptyState from '../components/ui/EmptyState';

export default function NotFoundPage() {
  return (
    <div className="card">
      <EmptyState
        icon={Compass}
        title="页面不存在"
        description="链接可能已失效，或路径输入有误。"
        action={
          <Link to="/" className="btn-primary">
            返回仪表盘
          </Link>
        }
      />
    </div>
  );
}
