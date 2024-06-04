import React from 'react';
import { Link } from 'react-router-dom';
import { Fragment } from 'react'
import { Menu, Transition } from '@headlessui/react'
import { EllipsisHorizontalIcon } from '@heroicons/react/20/solid'
import ClipboardCopyIcon from '../../components/ClipboardIcon';


interface Model {
    url: string;
    owner: string;
    name: string;
    description: string | undefined; // Update description to be optional to match the returned data
    visibility: string;
    github_url: string;
    paper_url: string | null; // Update paper_url to be nullable to match the returned data
    run_count: number;
}

const ModelLink: React.FC<{ model: Model }> = ({ model }) => {
  const copyToClipboard = () => {
    navigator.clipboard.writeText(`/${model?.name || ''}`);

  };

  return (
    <li key={model.name} className="overflow-hidden rounded-xl border-gray-200 bg-gray-100">
  <Link to={`/models/${model.name}`} className='mb-4'>
    <div className="flex items-center gap-x-4 bg-gray-500">
      <div className="flex-1">
        <div className="flex items-center text-sm text-gray-500">
          <h1 className="text-2xl font-bold text-gray-900">{model.name}</h1>
          <button
            type="button"
            className="ml-2 text-gray-500 hover:text-gray-700 focus:outline-none"
            onClick={copyToClipboard}
          >
            <ClipboardCopyIcon/>
          </button>
        </div>
        <div className="flex">
          <button
            type="button"
            className="rounded-md bg-indigo-50 px-2.5 py-1.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-100 mr-2"
          >
            {model.description || 'N/A'}
          </button>
          <button
            type="button"
            className="rounded-md bg-indigo-50 px-2.5 py-1.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-100 mr-2"
          >
            {model.run_count || 'N/A'}
          </button>
          <button
            type="button"
            className="rounded-md bg-indigo-50 px-2.5 py-1.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-100"
          >
            0.01$ / API Call
          </button>
        </div>
      </div>
      <div className="flex items-center">
        <button
          onClick={copyToClipboard}
          className="p-2.5 text-gray-400 hover:text-gray-500 focus:outline-none"
        >
        </button>
        <Menu as="div" className="relative">
          <Menu.Button className="-m-2.5 p-2.5 text-gray-400 hover:text-gray-500 focus:outline-none">
            <span className="sr-only">Open options</span>
            <EllipsisHorizontalIcon className="h-5 w-5" aria-hidden="true" />
          </Menu.Button>
          <Transition
            as={Fragment}
            enter="transition ease-out duration-100"
            enterFrom="transform opacity-0 scale-95"
            enterTo="transform opacity-100 scale-100"
            leave="transition ease-in duration-75"
            leaveFrom="transform opacity-100 scale-100"
            leaveTo="transform opacity-0 scale-95"
          >
            <Menu.Items className="absolute right-0 z-10 mt-0.5 w-32 origin-top-right rounded-md bg-white py-2 shadow-lg ring-1 ring-gray-900/5 focus:outline-none">
              <Menu.Item>
                {({ active }) => (
                  <a
                    href="#"
                    className={`${active ? 'bg-gray-50' : ''} block px-3 py-1 text-sm leading-6 text-gray-900`}
                  >
                    View<span className="sr-only">, {model.name}</span>
                  </a>
                )}
              </Menu.Item>
              <Menu.Item>
                {({ active }) => (
                  <a
                    href="#"
                    className={`${active ? 'bg-gray-50' : ''} block px-3 py-1 text-sm leading-6 text-gray-900`}
                  >
                    Edit<span className="sr-only">, {model.name}</span>
                  </a>
                )}
              </Menu.Item>
            </Menu.Items>
          </Transition>
        </Menu>
      </div>
    </div>
  </Link>
</li>
  );
}

export default ModelLink;