import ReactMarkdown from 'react-markdown';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';

interface MarkdownMessageProps {
  content: string;
}

const markdownComponents: Components = {
  // `inline` is provided by react-markdown at runtime; type as any to keep TS happy
  code({ inline, children, ...props }: any) {
    if (inline) {
      return (
        <code
          className="bg-base-200 rounded px-1 py-[2px] text-sm"
          {...props}
        >
          {children}
        </code>
      );
    }

    return (
      <pre className="bg-base-200 rounded p-3 overflow-x-auto whitespace-pre-wrap">
        <code className="text-sm" {...props}>
          {children}
        </code>
      </pre>
    );
  },
  a({ href, children, ...props }) {
    return (
      <a
        href={href}
        className="link"
        target="_blank"
        rel="noreferrer"
        {...props}
      >
        {children}
      </a>
    );
  },
};

export function MarkdownMessage({ content }: MarkdownMessageProps) {
  return (
    <ReactMarkdown
      className="chat-markdown"
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={markdownComponents}
      skipHtml
    >
      {content}
    </ReactMarkdown>
  );
}
