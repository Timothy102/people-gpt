import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import SideBarLayout from '../../layout/SideBarLayout';
import ModelUploadForm from '../../components/ModelUploadForm';
import UploadSuccess from '../../components/UploadSuccess';
import ModelLoadingAlert from '../../components/ModelLoading';
import { Fragment } from 'react'
import { Menu, Transition } from '@headlessui/react'
import { EllipsisHorizontalIcon } from '@heroicons/react/20/solid'
import ClipboardCopyIcon from '../../components/ClipboardIcon';


interface Model {
  model_id: number;
  name: string;
  description: string;
  output_path: string;
  category: string;
  long_description: string;
  use_case_category: string;
  charge_per_api_call: number;
  featured_image_url: string;
  user_id: number;
  creator_id: number;
}

const ModelPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [modelConfig, setModelConfig] = useState(null);
  const [modelFormSubmitted, setModelFormSubmitted] = useState(false);
  const [showModelLoadingAlert, setShowModelLoadingAlert] = useState(false);


  const navigate = useNavigate();

  useEffect(() => {
    fetchModels();
  }, []);



  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
    setFileUploaded(true);
  };

  const handleUpdateModelConfig = (config: any) => {
    // This function will be called from your ModelUploadForm component
    setModelConfig(config);
    // After receiving the model config, call your backend function with both file and config
    uploadModelAndConfig(file, config);
    setModelFormSubmitted(true);
    setShowModelLoadingAlert(true);
  };

  const handleUpload = (event: any) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile); // Store the uploaded file in the state
    setIsLoading(true);
  };


  const uploadModelAndConfig = async (file: any, config: any) => {
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('config', JSON.stringify(config));
  
      const response = await fetch('http://localhost:8000/deployments/upload_model_and_config/', {
        method: 'POST',
        body: formData,
      });
  
      if (response.ok) {
        //alert('Model and configuration uploaded successfully');
        setUploadSuccess(true);
        navigate("/");
        return <UploadSuccess />;
      } else {
        // Handle non-200 response
        const errorMessage = await response.text();
        throw new Error(`Failed to upload model and config: ${response.status} - ${errorMessage}`);
      }
    } catch (error) {
      console.error('Error uploading model and config:', error);
      alert('An error occurred while uploading the model and config');
    } finally {
      setIsLoading(false);
    }
  };


  const fetchModels = async () => {
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/models/models/');
      if (response.ok) {
        const data = await response.json();
        setModels(data as Model[]);
      } else {
        console.error('Failed to fetch models:', response.statusText);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SideBarLayout>
      <>
      <h1 className="text-2xl font-bold text-gray-900 mb-4">Get Started With Arcadia</h1>
      <h4 className="text-gray-900 mb-6 md:mb-8 lg:mb-10">
  Hey! Tim here. I built Arcadia out of my own necessity to charge other ML engineering teams access to proprietary ML models.
  I hope it helps you accomplish the same. Start by uploading your 1st ML Model on Arcadia!
</h4>
      <label
        htmlFor="uploadFile1"
        className="bg-white text-black text-base rounded w-80 h-52 flex flex-col items-center justify-center cursor-pointer border-2 border-gray-300 border-dashed mx-auto font-[sans-serif] mb-20"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="w-8 mb-2 fill-black" viewBox="0 0 32 32">
          <path
            d="M23.75 11.044a7.99 7.99 0 0 0-15.5-.009A8 8 0 0 0 9 27h3a1 1 0 0 0 0-2H9a6 6 0 0 1-.035-12 1.038 1.038 0 0 0 1.1-.854 5.991 5.991 0 0 1 11.862 0A1.08 1.08 0 0 0 23 13a6 6 0 0 1 0 12h-3a1 1 0 0 0 0 2h3a8 8 0 0 0 .75-15.956z"
            data-original="#000000"
          />
          <path
            d="M20.293 19.707a1 1 0 0 0 1.414-1.414l-5-5a1 1 0 0 0-1.414 0l-5 5a1 1 0 0 0 1.414 1.414L15 16.414V29a1 1 0 0 0 2 0V16.414z"
            data-original="#000000"
          />
        </svg>
        Upload file
        <input type="file" id="uploadFile1" className="hidden" onChange={handleFileChange} />
        <p className="text-xs text-gray-400 mt-2">PNG, JPG SVG, WEBP, and GIF are Allowed.</p>
      </label>
      <div>
    </div>

     {/* Model upload form */}
     { fileUploaded && !modelFormSubmitted && !uploadSuccess && <ModelUploadForm handleUpdateModelConfig={handleUpdateModelConfig} />}

    {/* Model loading alert */}
    {!uploadSuccess && showModelLoadingAlert && <ModelLoadingAlert onClose={() => setShowModelLoadingAlert(false)} />}


    <label/>
        <div>
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Play With Existing Models</h1>
          {models.length > 0 && (
            <div className="models-grid">
              {models.map((model) => (
                <div key={model.model_id} className="model-container">
                  {model.creator_id && (
                    <ModelLink key={model.model_id} model={model} />
                  )}
                </div>
              ))}
            </div>
          )}
          {isLoading && <p>Loading models...</p>}
          {!isLoading && models.length === 0 && <p>No models found.</p>}
        </div>
      </>
    </SideBarLayout>
  );
};

const ModelLink: React.FC<{ model: Model }> = ({ model }) => {
  const [username, setUsername] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean | null>(null);
  const copyToClipboard = () => {
    navigator.clipboard.writeText(`${username}/${model?.name || ''}`);
    setCopied(true);

    // Hide the success message after 3 seconds
    setTimeout(() => {
      setCopied(false);
    }, 2000);
  };

  const getUsername = async (userId: number) => {
    try {
      const response = await fetch(`http://localhost:8000/models/get_username_or_email?user_id=${userId}`);
      if (response.ok) {
        const userData = await response.json();
        return userData.username;
      } else {
        console.error('Failed to fetch username:', response.statusText);
        return ''; // Return an empty string if there's an error
      }
    } catch (error) {
      console.error('Error fetching username:', error);
      return ''; // Return an empty string if there's an error
    }
  };

  useEffect(() => {
    const fetchUsername = async () => {
      try {
        const fetchedUsername = await getUsername(model.creator_id);
        setUsername(fetchedUsername);
      } catch (error) {
        console.error('Error fetching username:', error);
      }
    };

    fetchUsername();
  }, [model.creator_id]);

  if (!username) return null;

  return (
    <li key={model.model_id} className="overflow-hidden rounded-xl border-gray-200 bg-gray-100">
  <Link to={`/models/${username}/${model.name}`} className='mb-4'>
    <div className="flex items-center gap-x-4 bg-gray-500">
      <div className="flex-1">
        <div className="flex items-center text-sm text-gray-500">
          <h1 className="text-2xl font-bold text-gray-900">{username}/{model.name}</h1>
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
            {model.category || 'N/A'}
          </button>
          <button
            type="button"
            className="rounded-md bg-indigo-50 px-2.5 py-1.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-100 mr-2"
          >
            {model.use_case_category || 'N/A'}
          </button>
          <button
            type="button"
            className="rounded-md bg-indigo-50 px-2.5 py-1.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-100"
          >
            {model.charge_per_api_call}$ / API Call
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

export default ModelPage;