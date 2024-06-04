import { Fragment } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { CheckIcon } from '@heroicons/react/24/outline';

interface PopupProps {
  isOpen: boolean;
  onClose: () => void;
  iframeText: string;
  scriptText: string;
}

export default function Popup({ isOpen, onClose, iframeText, scriptText }: PopupProps) {

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('Copied to clipboard!');
    }).catch((error) => {
      console.error('Error copying text:', error);
    });
  };

  return (
    <Transition.Root show={isOpen} as={Fragment}>
      <Dialog className="relative z-10" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
          <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
              enterTo="opacity-100 translate-y-0 sm:scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 translate-y-0 sm:scale-100"
              leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
            >
              <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-white px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-4xl sm:p-6">
                <div className="mt-4"> {/* Added mt-4 for margin-top */}
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-green-100 mb-4">
                    <CheckIcon className="h-6 w-6 text-green-600" aria-hidden="true" />
                  </div>
                  <div className="mt-3 text-center sm:mt-5">
                    <Dialog.Title as="h3" className="text-base font-semibold leading-6 text-gray-900">
                      Repository Selected
                    </Dialog.Title>
                    <div className="mt-4 flex justify-between space-x-4">
                      <div className="w-1/2">
                        <h3>HTML Code:</h3>
                        <pre className="whitespace-pre-wrap break-words bg-gray-100 p-2 rounded">{scriptText}</pre>
                        <button
                          onClick={() => copyToClipboard(scriptText)}
                          className="bg-blue-500 text-white px-4 py-2 rounded mt-2"
                        >
                          Copy HTML Script
                        </button>
                      </div>
                      <div className="w-1/2">
                        <h3>iFrame Code:</h3>
                        <pre className="whitespace-pre-wrap break-words bg-gray-100 p-2 rounded">{iframeText}</pre>
                        <button
                          onClick={() => copyToClipboard(iframeText)}
                          className="bg-blue-500 text-white px-4 py-2 rounded mt-2"
                        >
                          Copy iFrame
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-5 sm:mt-6">
                  <button
                    type="button"
                    className="inline-flex w-full justify-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                    onClick={onClose}
                  >
                    Close
                  </button>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}

