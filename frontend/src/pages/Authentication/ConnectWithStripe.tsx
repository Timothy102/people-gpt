import React, { useState } from 'react';
import Breadcrumb from '../../components/Breadcrumbs/Breadcrumb';
import SideBarLayout from '../../layout/SideBarLayout';

const ConnectToStripe: React.FC = () => {
  const [oauthLink, setOAuthLink] = useState<string>('');

  // Function to fetch the OAuth link from the backend
  const connectToStripe = async () => {
    try {
      const response = await fetch('https://arcadia-backend-m22shwtlv-teamrywaveios-projects.vercel.app/stripe/oauth'); 
      const data = await response.json()// Replace with your backend endpoint
      setOAuthLink(data.oauth_link);
      console.log('Stripe OAuth link:', data.oauth_link);

      window.location.href = data.oauth_link; // Redirect the user to the obtained OAuth link

    } catch (error) {
      console.error('Error connecting to Stripe:', error);
    }
  };

  return (
    <SideBarLayout>
      <Breadcrumb pageName="Connect with Stripe" />
      <div className="w-full border-stroke dark:border-strokedark xl:w-1/2 xl:border-l-2">
        <div className="w-full p-4 sm:p-12.5 xl:p-17.5">
          <span className="mb-1.5 block font-medium">Start for free</span>
          <h2 className="mb-9 text-2xl font-bold text-black dark:text-white sm:text-title-xl2">
            Connect with Stripe
          </h2>

          {/* Connect with Stripe button */}
          <div className="mb-5">
            <button
              type="button"
              onClick={connectToStripe} // Call connectToStripe when the button is clicked
              className="w-full cursor-pointer rounded-lg border border-primary bg-primary p-4 text-white transition hover:bg-opacity-90"
            >
              Connect with Stripe
            </button>
          </div>

          <div className="mt-6 text-center">
            <p>
              Already have a Stripe account?{' '}
              {oauthLink && (
                <a href={oauthLink} className="text-primary" target="_blank" rel="noopener noreferrer">
                  Log In
                </a>
              )}
            </p>
          </div>
        </div>
      </div>
    </SideBarLayout>
  );
};

export default ConnectToStripe;
