import React, { useState, useEffect } from 'react';
import SideBarLayout from '../../layout/SideBarLayout';

const ModelUsage: React.FC = () => {
  const [userSpenditure, setUserSpenditure] = useState<any>(null);


  useEffect(() => {
    const fetchUserSpenditure = async () => {
     
      try {
        const response = await fetch('http://localhost:8000/models/get_user_spenditure/1', {
          method: 'GET',
        });
        if (response.ok) {
          const data = await response.json();
          setUserSpenditure(data);
        } else {
          console.error('Failed to fetch user expenditure:', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching user expenditure:', error);
      }
    };

    fetchUserSpenditure();
  }, []);

  if (!userSpenditure) {
    return <p>Loading...</p>;
  }

  return (
    <SideBarLayout>
    <div>
      <h2>User Expenditure</h2>
      <p>User ID: {userSpenditure.user_id}</p>
      <p>Username: {userSpenditure.username}</p>
      <p>Email: {userSpenditure.email}</p>
      <p>Total Amount Due: {userSpenditure.total_amount_due}</p>

      <h3>Models</h3>
      {/* <ul>
        {userSpenditure.models.map((model: any, index: number) => (
          <li key={index}>
            <p>Name: {model.name}</p>
            <p>Description: {model.description}</p>
          </li>
        ))}
      </ul> */}

      <h3>Model Usages</h3>
      {/* <ul>
        {userSpenditure.model_usages.map((modelUsage: any, index: number) => (
          <li key={index}>
            <p>Usage Count: {modelUsage.usage_count}</p>
            <p>Amount Due This Month: {modelUsage.amount_due_this_month}</p>
          </li>
        ))}
      </ul> */}
    </div>
    </SideBarLayout>
  );
};

export default ModelUsage;