import React from 'react';

interface ButtonProps {
  text: string;
  onClick?: () => void;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
}

const Button: React.FC<ButtonProps> = ({ text, onClick, size = 'md' }) => {
  const baseClass = "rounded bg-indigo-500 text-white shadow-sm hover:bg-indigo-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500";
  let sizeClass = '';

  switch (size) {
    case 'xs':
      sizeClass = 'px-2 py-1 text-xs font-semibold';
      break;
    case 'sm':
      sizeClass = 'px-2 py-1 text-sm font-semibold';
      break;
    case 'md':
      sizeClass = 'rounded-md px-2.5 py-1.5 text-sm font-semibold';
      break;
    case 'lg':
      sizeClass = 'rounded-md px-3 py-2 text-sm font-semibold';
      break;
    case 'xl':
      sizeClass = 'rounded-md px-3.5 py-2.5 text-sm font-semibold';
      break;
    default:
      sizeClass = 'rounded-md px-2.5 py-1.5 text-sm font-semibold';
  }

  return (
    <button
      type="button"
      className={`${baseClass} ${sizeClass}`}
      onClick={onClick}
    >
      {text}
    </button>
  );
};

export default Button;
