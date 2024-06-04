import { useEffect, useState } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';

import Loader from './common/Loader';
import PageTitle from './components/PageTitle';
import NotFoundPage from './components/404';
import CreatePage from './pages/Dashboard/Create'
import Chatbot from './pages/Dashboard/ModelDetails';

function App() {
  const [loading, setLoading] = useState<boolean>(true);
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000);
  }, []);

  return loading ? (
    <Loader />
  ) : (
    <>
      <Routes>
        <Route
          path="/"
          element={
            <>
              <PageTitle title="Find People" />
              <CreatePage />
            </>
          }
        />
        <Route path="/2e2ed734-9519-48fa-adb4-8290287b0697/chat" element={<Chatbot />} />
        <Route
          path="/404"
          element={
            <>
              <PageTitle title="404 - Page Not Found" />
              <NotFoundPage />
            </>
          }
        />
      </Routes>
    </>
  );
}

export default App;
