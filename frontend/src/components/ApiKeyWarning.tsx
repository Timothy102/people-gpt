import { ExclamationTriangleIcon } from '@heroicons/react/20/solid'

export default function ApiKeyWarning() {
  return (
    <div className="rounded-md bg-yellow-50 p-4 mb-10">
      <div className="flex">
        <div className="flex-shrink-0">
          <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400" aria-hidden="true" />
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-yellow-800">Attention needed</h3>
          <div className="mt-2 text-sm text-yellow-700">
            <p>
            Warning! Do not share your API keys with others, or expose it in the browser or other client-side code. In order to protect the security of your account, Arcadia may also automatically disable any API key that we've found has leaked publicly. You're the one getting billed.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
