import { useState } from 'react';
import SideBarLayout from '../layout/SideBarLayout';
import { UserCircleIcon } from '@heroicons/react/24/outline';

const Profile = () => {
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [profileImage, setProfileImage] = useState<File | null>(null); // Change the type to File | null
  const [successMessage, setSuccessMessage] = useState('');

  const [about, setAbout] = useState('');

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      setProfileImage(file);
    }
  };


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('about', about);
      if (profileImage) {
        formData.append('profileImage', profileImage);
      }

      // Call API to update user profile
      const response = await fetch(`http://localhost:8000/models/change_profile/1`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: formData,
      });

      if (response.ok) {
        setSuccessMessage("Profile Updated Successfully.")
      }
      // Handle success scenario, e.g., show success message
    } catch (error) {
      if (typeof error === 'string') {
        setError(error);
      } else {
        setError('An error occurred.');
      }
    }
  };

  // const handleChangeUsername = async () => {
  //   try {
  //     const response = await fetch(`http://localhost:8000/models/change_username/1?username=${username}`, {
  //       method: 'PUT',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({ username: username }),
  //     });

  //     if (response.ok) {
  //       const data = await response.json();
  //       setSuccessMessage(`Username changed to ${data.username}`);
  //     } else {
  //       const data = await response.json();
  //       setError(data.detail);
  //     }
  //   } catch (error) {
  //     console.error('Error changing username:', error);
  //     setError('Failed to change username. Please try again later.');
  //   }
  // };

  return (
    <SideBarLayout>
      <form onSubmit={handleSubmit}>
        <div className="space-y-12">
          <div className="border-b border-gray-900/10 pb-12">
            <h2 className="text-base font-semibold leading-7 text-gray-900">Profile</h2>
            <p className="mt-1 text-sm leading-6 text-gray-600">
              This information will be displayed publicly so be careful what you share.
            </p>
  
            {/* Username input */}
            <div className="mt-10 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-6">
              <div className="sm:col-span-4">
                <label htmlFor="username" className="block text-sm font-medium leading-6 text-gray-900">
                  Username
                </label>
                <div className="mt-2">
                  <div className="flex rounded-md shadow-sm ring-1 ring-inset ring-gray-300 focus-within:ring-2 focus-within:ring-inset focus-within:ring-indigo-600 sm:max-w-md">
                    <span className="flex select-none items-center pl-3 text-gray-500 sm:text-sm">arcadia.ai/</span>
                    <input
                      type="text"
                      name="username"
                      id="username"
                      autoComplete="username"
                      className="block flex-1 border-0 bg-transparent py-1.5 pl-1 text-gray-900 placeholder:text-gray-400 focus:ring-0 sm:text-sm sm:leading-6"
                      placeholder="janesmith"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                    />
                  </div>
                </div>
              </div>
            </div>
  
            {/* About section */}
            <div className="mt-10 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-6">
              <div className="col-span-full">
                <label htmlFor="about" className="block text-sm font-medium leading-6 text-gray-900">
                  About
                </label>
                <div className="mt-2">
                  <p className="mt-3 text-sm leading-6 text-gray-600">Write a few sentences about yourself.</p>
                  <textarea
                    id="about"
                    name="about"
                    rows={3}
                    className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                    value={about}
                    onChange={(e) => setAbout(e.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="col-span-full">
          <label htmlFor="photo" className="block text-sm font-medium leading-6 text-gray-900">
            Photo
          </label>
          <div className="mt-2 flex items-center gap-x-3">
            <UserCircleIcon className="h-12 w-12 text-gray-300" aria-hidden="true" />
            <label htmlFor="profileImage" className="rounded-md bg-white px-2.5 py-1.5 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 cursor-pointer">
              Change
              <input type="file" id="profileImage" accept="image/*" className="hidden" onChange={handleImageChange} />
            </label>
          </div>
        </div>
  
            {/* Error message */}
            {error && <div className="text-red-600">{error}</div>}
  
            {/* Success message */}
            {successMessage && <div className="text-green-600">{successMessage}</div>}
  
            {/* Save button */}
            <div className="mt-6 flex items-center justify-end gap-x-6">
              <button
                type="submit"
                className="rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      </form>
    </SideBarLayout>
  );  
};

export default Profile;