import { useState } from 'react';

const Chatbot = () => {
    const [chatbotHeight, setChatbotHeight] = useState(0);
    const [buttonText, setButtonText] = useState('▼');

    const toggleChatbot = () => {
        if (chatbotHeight === 0) {
            setChatbotHeight(500);
            setButtonText('▲');
        } else {
            setChatbotHeight(0);
            setButtonText('▼');
        }
    };

    return (
        <div>
            <button
                id="chatbot_button"
                onClick={toggleChatbot}
                style={{
                    position: 'fixed',
                    bottom: '20px',
                    right: '20px',
                    width: '50px',
                    height: '50px',
                    border: 'none',
                    borderRadius: '50%',
                    backgroundColor: '#007bff',
                    color: '#fff',
                    fontSize: '24px',
                    cursor: 'pointer',
                    zIndex: '9999',
                    margin: '0',
                    padding: '0'
                }}
            >
                {buttonText}
            </button>

            <div
                id="chatbot_window"
                style={{
                    position: 'fixed',
                    bottom: '20px',
                    right: '20px',
                    width: '350px',
                    height: `${chatbotHeight}px`,
                    border: '1px solid #ccc',
                    borderRadius: '10px',
                    overflow: 'hidden',
                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
                    transition: 'height 0.3s ease',
                    zIndex: '9998'
                }}
            >
                {/* Embedded chatbot window */}
                {chatbotHeight !== 0 && (
                    <iframe
                        id="chatbot_iframe"
                        src="http://localhost:3000"
                        style={{
                            border: 'none',
                            width: '100%',
                            height: '100%'
                        }}
                    />
                )}
            </div>
        </div>
    );
};

export default Chatbot;