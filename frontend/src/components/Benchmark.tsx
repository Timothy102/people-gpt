import { StarIcon } from '@heroicons/react/20/solid';

interface BenchmarkButtonProps {
  totalCallsMade: number | null; // Accepts null if the value is not yet available
}

const BenchmarkButton: React.FC<BenchmarkButtonProps> = ({ totalCallsMade }) => {
  return (
    <span className="isolate inline-flex rounded-md shadow-sm">
      <button
        type="button"
        className="relative inline-flex items-center gap-x-1.5 rounded-l-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10"
      >
        <StarIcon className="-ml-0.5 h-5 w-5 text-gray-400" aria-hidden="true" />
        Total API Calls Made
      </button>
      <button
        type="button"
        className="relative -ml-px inline-flex items-center rounded-r-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10"
      >
        {totalCallsMade !== null ? totalCallsMade.toLocaleString() : 'Loading...'}
      </button>
    </span>
  );
};

export default BenchmarkButton;
