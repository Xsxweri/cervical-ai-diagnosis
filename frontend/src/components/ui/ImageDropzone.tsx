import { useCallback, useRef, useState } from 'react';
import { UploadCloud, X } from 'lucide-react';

interface Props {
  onFile: (file: File) => void;
  accept?: string;
  maxSizeMB?: number;
  hint?: string;
}

export default function ImageDropzone({
  onFile,
  accept = 'image/*',
  maxSizeMB = 30,
  hint = '支持 JPG / PNG / BMP，最大 30MB',
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) {
        setError('请选择图片文件');
        return;
      }
      if (file.size > maxSizeMB * 1024 * 1024) {
        setError(`文件超过 ${maxSizeMB}MB 限制`);
        return;
      }
      setError(null);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      onFile(file);
    },
    [maxSizeMB, onFile]
  );

  const reset = useCallback(() => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  }, []);

  return (
    <div>
      <div
        className={`border-2 border-dashed rounded-xl transition-all cursor-pointer flex flex-col items-center justify-center p-10 text-center ${
          dragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-200 bg-gray-50/50 hover:border-primary-400 hover:bg-primary-50/40'
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
      >
        {previewUrl ? (
          <div className="relative">
            <img
              src={previewUrl}
              alt="预览"
              className="max-h-64 rounded-lg object-contain shadow-sm"
            />
            <button
              className="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-gray-800/80 text-white flex items-center justify-center hover:bg-gray-900"
              onClick={(e) => {
                e.stopPropagation();
                reset();
              }}
              aria-label="移除图片"
            >
              <X size={16} />
            </button>
          </div>
        ) : (
          <>
            <div className="w-14 h-14 rounded-full bg-primary-50 text-primary-500 flex items-center justify-center mb-3">
              <UploadCloud size={26} />
            </div>
            <p className="font-medium text-gray-700">点击或拖拽图片到此处</p>
            <p className="text-sm text-gray-400 mt-1">{hint}</p>
          </>
        )}
      </div>

      {error && (
        <p className="mt-2 text-sm text-danger-600">{error}</p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}