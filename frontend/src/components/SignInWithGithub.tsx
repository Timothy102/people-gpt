import { useState, useEffect, ChangeEvent } from 'react';
import Popup from './Popup'; // Adjust the import path according to your directory structure

interface Repo {
  id: number;
  full_name: string;
}

const SignInWithGitHub = () => {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<string>('');
  const [showPopup, setShowPopup] = useState<boolean>(false);

  useEffect(() => {
    const getQueryParam = (name: string) => {
      const urlParams = new URLSearchParams(window.location.search);
      return urlParams.get(name);
    };

    const code = getQueryParam('code');
    console.log("Authorization code:", code);

    if (code) {
      fetch(`http://localhost:8000/callback?code=${code}`)
      .then(response => {
          if (!response.ok) {
              throw new Error('Network response was not ok');
          }
          return response.json();
      })
      .then(data => {
          console.log('Access Token:', data.access_token);
          console.log('Callback Response:', data);
  
          fetch(`https://api.github.com/users/timothy102/repos`, {
              headers: {
                  'Authorization': `token ${data.access_token}`
              }
          })
          .then(response => response.json())
          .then((reposData: Repo[]) => {
              console.log('User Repos:', reposData);
              setRepos(reposData);
          })
          .catch(error => {
              console.error('Error fetching user repositories:', error);
          });
      })
      .catch(error => {
          console.error('Error sending authorization code to backend:', error);
      });
  }
  }, []);

  const handleSignIn = () => {
    try {
      window.location.href = `https://github.com/login/oauth/authorize?client_id=Ov23liR90U1WLoBqgJQq&redirect_uri=http://localhost:5173&scope=user`;
    } catch (error) {
      console.error('Error signing in with GitHub:', error);
    }
  };

  const handleRepoChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const selectedRepo = event.target.value;
    setSelectedRepo(selectedRepo);

    // Send the selected repository to the backend
    fetch(`http://localhost:8000/selected_repo?repo=${selectedRepo}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    .then(response => response.json())
    .then(data => {
      console.log('Backend response:', data);
    })
    .catch(error => {
      console.error('Error sending selected repository to backend:', error);
    });
  };

  const handleGoClick = () => {
    setShowPopup(true);
  };

  const closePopup = () => {
    setShowPopup(false);
  };

  const iframeText = `<iframe
src="https://www.chatbase.co/chatbot-iframe/kHL3KMlAq6Vo-_ICuHXHn"
title="Chatbot"
width="100%"
style="height: 100%; min-height: 700px"
frameborder="0"
></iframe>`;

  const scriptText = `<!DOCTYPE html>
  <html lang="en">
  
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Embedded Chatbot</title>
      <style>
          body {
              margin: 0;
              padding: 0;
              font-family: Arial, sans-serif;
              position: relative;
              min-height: 100vh;
          }
  
          #chatbot_button {
              position: fixed;
              bottom: 20px;
              right: 20px;
              width: 50px;
              height: 50px;
              border: none;
              border-radius: 50%;
              background-color: #007bff;
              color: #fff;
              font-size: 24px;
              cursor: pointer;
              z-index: 9999;
              margin: 0; /* Remove margin */
              padding: 0; /* Remove padding */
          }
  
          #chatbot_window {
              position: fixed;
              bottom: 20px;
              right: 20px;
              width: 350px;
              height: 500px;
              border: 1px solid #ccc;
              border-radius: 10px;
              overflow: hidden;
              box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
              transition: height 0.3s ease;
              z-index: 9998;
          }
  
          #chatbot_iframe {
              border: none;
              width: 100%;
              height: 100%;
          }
      </style>
  </head>
  
  <body>
      <!-- Button to toggle the chatbot window -->
      <button id="chatbot_button" onclick="toggleChatbot()">
          &#x25BC;
      </button>
  
      <!-- Embedded chatbot window -->
      <div id="chatbot_window" style="height: 0;">
          <iframe id="chatbot_iframe" src="http://localhost:3000/2e2ed734-9519-48fa-adb4-8290287b0697/chat"></iframe>
      </div>
  
      <!-- JavaScript code for toggling the chatbot window -->
      <script>
          function toggleChatbot() {
              const chatbotWindow = document.getElementById('chatbot_window');
              const chatbotButton = document.getElementById('chatbot_button');
  
              if (chatbotWindow.style.height === '0px' || chatbotWindow.style.height === '') {
                  chatbotWindow.style.height = '500px';
                  chatbotButton.innerHTML = '&#x25B2;'; // Change button text to up arrow
              } else {
                  chatbotWindow.style.height = '0';
                  chatbotButton.innerHTML = '&#x25BC;'; // Change button text to down arrow
              }
          }
      </script>
  </body>
  
  </html>`;

  return (
    <div>
      <button onClick={handleSignIn} className="bg-blue-500 text-black px-4 py-2 rounded mb-10">
        Sign In with GitHub
      </button>
      {repos.length > 0 ? (
        <div>
          <label htmlFor="repoSelect">Select a Repository: </label>
          <select id="repoSelect" value={selectedRepo} onChange={handleRepoChange}>
            <option value="">Select a repository</option>
            {repos.map(repo => (
              <option key={repo.id} value={repo.full_name}>{repo.full_name}</option>
            ))}
          </select>
          {selectedRepo && (
            <button onClick={handleGoClick} className="bg-green-500 text-white px-4 py-2 rounded ml-2">
              Go
            </button>
          )}
        </div>
      ) : (
        <p>Loading repositories...</p>
      )}
      <Popup
        isOpen={showPopup}
        onClose={closePopup}
        iframeText={iframeText}
        scriptText={scriptText}
      />
    </div>
  );
};

export default SignInWithGitHub;