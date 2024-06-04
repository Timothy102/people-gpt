import SideBarLayout from '../../layout/SideBarLayout';
import { useState } from 'react';
import Table from '../../components/Table';

const CreatePage: React.FC = () => {
  const [copiedText, setCopiedText] = useState('');
  const [inputText, setInputText] = useState('');
  const [showTable, setShowTable] = useState(false);
  const handleCopyText = (text: string) => {
    setCopiedText(text);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
  };

  const handleGoButtonClick = () => {
    // Implement your logic here when the "Go!" button is clicked
    console.log("Go button clicked!");
    setShowTable(true);
  };

  // Array of randomly generated roles with emojis
  const roles = [
    "👩‍💻 Climate-focused angel investor in Europe",
    "🚀 Blockchain developer for decentralized finance",
    "🌐 AI specialist for sustainable energy solutions",
    "💼 UX designer for inclusive technology products",
  ];

  return (
    <SideBarLayout>
      <div className='mb-10 text-center'>
        <h1 className="text-4xl font-bold text-gray-900 mb-8">Find out who you're looking for!</h1>
        
        {/* Centered Textarea */}
        <div className="mx-auto w-full max-w-xl bg-blue-100 rounded-lg p-6 mb-4">
          <textarea
            className="w-full h-40 bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-400 resize-none mb-4"
            placeholder="Enter your search text..."
            value={inputText}
            onChange={handleInputChange}
          />
          {/* Go Button */}
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 focus:outline-none"
            onClick={handleGoButtonClick}
          >
            Go!
          </button>
        </div>

        {/* Display Table conditionally */}
        {showTable && <Table />}

        {/* 2x2 Grid of White Rectangles */}
        {!showTable && (
          <div className="grid grid-cols-2 gap-4 mt-4">
            {roles.map((role, index) => (
              <div
                key={index}
                className="bg-white shadow-md rounded-lg p-6 cursor-pointer"
                onClick={() => handleCopyText(role)}
              >
                {role}
              </div>
            ))}
          </div>
        )}
      </div>
    </SideBarLayout>
  );
};

export default CreatePage;
