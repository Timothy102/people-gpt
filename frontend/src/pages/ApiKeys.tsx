import { useEffect, useState } from 'react';
import SideBarLayout from '../layout/SideBarLayout';
import ApiKeyWarning from '../components/ApiKeyWarning';

interface ApiKey {
  id: number;
  api_key: string;
  created_at: string;
  user_id: number;
}

const ApiKeysComponent = () => {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);

  useEffect(() => {
    const fetchApiKeys = async () => {
      try {
        const userId = 1;
        const response = await fetch(`http://localhost:8000/auth/api_keys/${userId}`);
        if (response.ok) {
          const data = await response.json();
          setApiKeys(data);
        } else {
          console.error('Failed to fetch API keys');
        }
      } catch (error) {
        console.error('Error fetching API keys:', error);
      }
    };

    fetchApiKeys();
  }, []);

  return (
    <SideBarLayout>
    <ApiKeyWarning></ApiKeyWarning>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                ID
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                API Key
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Created At
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {apiKeys.map(apiKey => (
              <tr key={apiKey.id}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{apiKey.id}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{apiKey.api_key}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{apiKey.created_at}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{apiKey.user_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SideBarLayout>
  );
};

export default ApiKeysComponent;